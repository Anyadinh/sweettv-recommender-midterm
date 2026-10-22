#!/usr/bin/env python3
"""
Two-Layer рекомендатель для SweetTV: генерация кандидатов + LightGBM-ранжирование.

Слой 1 (candidate generation) — до 200 кандидатов на юзера из трёх источников:
  * Top-50 по истории юзера в train (важно: >50% ground truth — это повторы)
  * Top-100 по скору ALS (в том числе новые для юзера программы)
  * Top-50 глобально популярных (перекрытие холодных юзеров)

Слой 2 (LightGBM) — бинарный классификатор "попадёт ли кандидат в Топ-5 юзера",
обученный на парах (юзер, кандидат) с целевым периодом ПОСЛЕ окна истории.

Протокол (честный, без подглядывания):
  fold "inner" : история = недели 11–15, метки = недели 16–20  -> обучение ранкера
  fold "valid" : история = недели 11–20, метки = недели 21–30  -> оценка MAP@5
  финал        : история = недели 11–30, метки = 16–20 и 21–30 -> сабмит на 31–42

Запуск:
  python3 two_layer.py --mode validation
  python3 two_layer.py --mode submit
  python3 two_layer.py --mode both
"""

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import duckdb
import lightgbm as lgb
import scipy.sparse as sp
from lightfm import LightFM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recsys_models as R

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


# ============================================================ подготовка данных

def build_rich_agg(con, alias):
    """Обогащённый агрегат (user, show) из таблицы {alias}_int."""
    con.execute(f"""
      CREATE OR REPLACE TABLE {alias}_r AS
      SELECT user_id, tv_show_id,
             count(*)::BIGINT         AS n_airings,
             max(screen_time)         AS max_screen_time,
             sum(screen_time)         AS sum_screen_time,
             sum(overlap_seconds)     AS total_overlap,
             max(session_start)       AS last_watch,
             mode(channel_id)         AS channel_id,
             any_value(tv_show_title) AS tv_show_title,
             any_value(tv_show_category) AS tv_show_category,
             any_value(tv_show_genre_1)  AS tv_show_genre_1,
             any_value(tv_show_genre_2)  AS tv_show_genre_2,
             any_value(tv_show_genre_3)  AS tv_show_genre_3
      FROM {alias}_int GROUP BY user_id, tv_show_id
    """)


def period_meta(con, sched_alias, extra_where=""):
    """Метаданные программ, выходящих в конкретный период: название, категория, канал."""
    rows = con.execute(f"""
        SELECT tv_show_id, tv_show_title, tv_show_category, channel_id
        FROM {sched_alias} WHERE tv_show_id > 0 {extra_where}
    """).fetchall()
    title, cat, chan_cnt = {}, {}, {}
    for sid, t, c, ch in rows:
        sid = int(sid)
        if t and sid not in title:
            title[sid] = t
        if c and sid not in cat:
            cat[sid] = c
        d = chan_cnt.setdefault(sid, {})
        d[int(ch)] = d.get(int(ch), 0) + 1
    chan = {sid: max(d.items(), key=lambda kv: (kv[1], -kv[0]))[0] for sid, d in chan_cnt.items()}
    return title, cat, chan


def global_category_map(con):
    rows = con.execute("SELECT tv_show_id, tv_show_category FROM meta_by_id").fetchall()
    cats = sorted({str(c) for _, c in rows if c})
    code = {c: i + 1 for i, c in enumerate(cats)}   # 0 = "неизвестно"
    out = {}
    for sid, c in rows:
        if c:
            out[int(sid)] = code[str(c)]
    return out, code


# ============================================================ слой 1: кандидаты

def train_als(con, agg_table, value_col="n_airings", factors=8, iterations=20,
              regularization=10.0, seed=42):
    """ALS на окне истории. Возвращает (model, items, user_index, item_index)."""
    import scipy.sparse as sp
    from implicit.als import AlternatingLeastSquares

    col = "n_airings" if value_col == "airings" else "max_screen_time"
    rows = con.execute(f"""
        SELECT user_id, tv_show_id, {col} FROM {agg_table}
        WHERE {col} > 0
    """).fetchall()
    users = sorted({int(r[0]) for r in rows})
    items = sorted({int(r[1]) for r in rows})
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {it: i for i, it in enumerate(items)}
    r = np.array([uidx[int(x[0])] for x in rows], dtype=np.int32)
    c = np.array([iidx[int(x[1])] for x in rows], dtype=np.int32)
    v = np.array([float(x[2]) for x in rows], dtype=np.float32)
    matrix = sp.csr_matrix((v, (r, c)), shape=(len(users), len(items)))
    model = AlternatingLeastSquares(factors=factors, iterations=iterations,
                                    regularization=regularization,
                                    random_state=seed, use_gpu=False)
    model.fit(matrix, show_progress=False)
    return model, items, uidx, iidx


def train_lightfm_biases(con, agg_table, users, items, uidx, iidx,
                          epochs=30, components=20, seed=42):
    """LightFM (loss='warp') на окне истории -> user_biases / item_biases.

    Биасы — это log-склонности: item_bias ~ популярность программы,
    user_bias ~ активность юзера. Обучаем на том же окне, что и остальные признаки,
    чтобы не было утечки из будущего.
    """
    rows = con.execute(f"""
        SELECT user_id, tv_show_id, n_airings FROM {agg_table} WHERE n_airings > 0
    """).fetchall()
    r = np.array([uidx[int(x[0])] for x in rows], dtype=np.int32)
    c = np.array([iidx[int(x[1])] for x in rows], dtype=np.int32)
    v = np.array([float(x[2]) for x in rows], dtype=np.float32)
    mat = sp.csr_matrix((v, (r, c)), shape=(len(users), len(items)))
    model = LightFM(loss="warp", no_components=components, random_state=seed)
    model.fit(mat, epochs=epochs, num_threads=1)
    ue, ub = model.get_user_representations()
    ie, ib = model.get_item_representations()
    ue = np.asarray(ue, dtype=np.float64)
    ie = np.asarray(ie, dtype=np.float64)
    ub = np.asarray(ub).ravel()
    ib = np.asarray(ib).ravel()
    user_bias = {int(users[i]): float(ub[i]) for i in range(len(users))}
    item_bias = {int(items[i]): float(ib[i]) for i in range(len(items))}
    return user_bias, item_bias, ue, ie


def popular_order(con, agg_table, universe):
    """Глобальный порядок популярности в окне истории (только программы из universe)."""
    rows = con.execute(f"""
        SELECT tv_show_id, sum(n_airings) f FROM {agg_table}
        GROUP BY tv_show_id ORDER BY f DESC, tv_show_id ASC
    """).fetchall()
    return [int(x[0]) for x in rows if int(x[0]) in universe]


def gen_candidates(con, hist_table, model, items, uidx, users, universe,
                   pop_order, k_hist=50, k_als=100, k_pop=50, cap=200):
    """Кандидаты на юзера: история + ALS + популярные -> union, dedup, cap."""
    # --- Top-50 по истории юзера
    rows = con.execute(f"SELECT user_id, tv_show_id, n_airings FROM {hist_table}").fetchall()
    hist = {}
    for u, it, n in rows:
        hist.setdefault(int(u), []).append((int(n), int(it)))
    for u in hist:
        hist[u] = [it for _, it in sorted(hist[u], key=lambda x: (-x[0], x[1]))[:k_hist]]

    # --- Top-100 по ALS
    uf = np.asarray(model.user_factors)
    itf = np.asarray(model.item_factors)
    items_arr = np.asarray(items, dtype=np.int64)
    mask = np.array([it in universe for it in items], dtype=bool)
    k_als_eff = min(k_als, len(items) - 1)
    als_top, als_rank = {}, {}
    for u in users:
        if u not in uidx:
            als_top[u] = []
            continue
        s = uf[uidx[u]] @ itf.T
        s[~mask] = -np.inf
        idx = np.argpartition(-s, k_als_eff)[:k_als_eff]
        idx = idx[np.isfinite(s[idx])]
        order = np.lexsort((items_arr[idx], -s[idx]))
        ranked = [int(items_arr[idx[i]]) for i in order]
        als_top[u] = ranked
        als_rank[u] = {it: i + 1 for i, it in enumerate(ranked)}

    pop_top = [it for it in pop_order[:k_pop] if it in universe]

    cands, flags = {}, {}
    for u in users:
        h = [it for it in hist.get(u, []) if it in universe]
        a = als_top.get(u, [])
        merged, seen = [], set()
        for it in h + a + pop_top:
            if it not in seen:
                seen.add(it)
                merged.append(it)
        cands[u] = merged[:cap]
        flags[u] = {it: (it in set(h), it in als_rank.get(u, {}), als_rank.get(u, {}).get(it, 0))
                    for it in cands[u]}
    return cands, flags


# ============================================================ слой 2: признаки

FEATURE_NAMES = [
    "user_watch_count", "user_screen_time_max", "user_screen_time_sum", "recency_days",
    "als_score", "als_score_rel", "als_rank", "popularity", "n_users_item",
    "user_watch_share", "user_st_share", "popularity_share",
    "is_watched", "from_history", "from_als", "from_pop",
    "user_cat_affinity", "user_chan_affinity",
    "category", "channel",
    "n_airings_period", "period_air_share", "hist_rank",
    "user_bias", "item_bias", "lfm_score",
]
CAT_FEATURES = ["category", "channel"]
CAT_IDX = [FEATURE_NAMES.index(c) for c in CAT_FEATURES]
BIAS_IDX = [FEATURE_NAMES.index("user_bias"), FEATURE_NAMES.index("item_bias")]
LFM_IDX = BIAS_IDX + [FEATURE_NAMES.index("lfm_score")]


def build_features(con, hist_table, users, cands, flags, model, items, uidx, iidx,
                   universe, cat_code, chan_code, ref_date, pop_global, item_users,
                   pop_set, period_airings, user_bias=None, item_bias=None,
                   lfm_emb=None):
    """Матрица признаков по парам (юзер, кандидат)."""
    items_arr = np.asarray(items, dtype=np.int64)
    uf = np.asarray(model.user_factors)
    itf = np.asarray(model.item_factors)

    # --- статистика по юзерам (окно истории)
    rows = con.execute(f"""
        SELECT user_id, tv_show_id, n_airings, max_screen_time, sum_screen_time,
               last_watch, channel_id
        FROM {hist_table}
    """).fetchall()
    hist_lookup = {}
    user_hist_rank = {}
    user_total_air = {}
    user_total_st = {}
    user_cat = {}
    user_chan = {}
    for u, it, n, mx, sm, lw, ch in rows:
        u, it = int(u), int(it)
        hist_lookup[(u, it)] = (int(n), float(mx), float(sm), lw)
        user_hist_rank.setdefault(u, []).append((int(n), it))
        user_total_air[u] = user_total_air.get(u, 0) + int(n)
        user_total_st[u] = user_total_st.get(u, 0.0) + float(sm)
        c = cat_code.get(it, -1)
        if c >= 0:
            user_cat.setdefault(u, {})
            user_cat[u][c] = user_cat[u].get(c, 0) + int(n)
        ch_code = chan_code.get(it, -1)
        if ch_code >= 0:
            user_chan.setdefault(u, {})
            user_chan[u][ch_code] = user_chan[u].get(ch_code, 0) + int(n)

    user_bias = user_bias or {}
    item_bias = item_bias or {}
    ue_l, ie_l = (lfm_emb if lfm_emb else (None, None))
    for u, lst in user_hist_rank.items():
        lst.sort(key=lambda x: (-x[0], x[1]))
        user_hist_rank[u] = {it: i + 1 for i, (_, it) in enumerate(lst)}
    total_pop = sum(pop_global.values()) or 1
    total_period = sum(period_airings.values()) or 1

    # --- скоринги ALS по юзерам
    user_scores = {}
    for u in users:
        if u in uidx:
            user_scores[u] = (uf[uidx[u]] @ itf.T).astype(np.float64)

    X_rows, y_groups = [], []
    for u in users:
        cl = cands.get(u, [])
        if not cl:
            y_groups.append((u, 0, 0))
            continue
        srow = user_scores.get(u)
        max_abs = float(np.abs(srow).max()) if srow is not None and srow.size else 1.0
        max_abs = max_abs if max_abs > 0 else 1.0
        tot_air = max(user_total_air.get(u, 0), 1)
        tot_st = max(user_total_st.get(u, 0.0), 1e-9)
        ucat = user_cat.get(u, {})
        uchan = user_chan.get(u, {})
        for it in cl:
            h = hist_lookup.get((u, it))
            n_air, mx_st, sum_st, last_watch = h if h else (0, 0.0, 0.0, None)
            if last_watch is not None:
                rec = (ref_date - last_watch).days
            else:
                rec = -1
            if srow is not None and it in iidx:
                als = float(srow[iidx[it]])
            else:
                als = 0.0
            f_hist, f_als, a_rank = flags[u][it]
            pop = float(pop_global.get(it, 0))
            cc = cat_code.get(it, 0)
            chc = chan_code.get(it, 0)
            X_rows.append([
                n_air, mx_st, sum_st, rec,
                als, als / max_abs, a_rank, pop, float(item_users.get(it, 0)),
                n_air / tot_air, sum_st / tot_st, pop / total_pop,
                1 if n_air > 0 else 0, 1 if f_hist else 0, 1 if f_als else 0,
                1 if it in pop_set else 0,
                ucat.get(cc, 0) / tot_air,
                uchan.get(chc, 0) / tot_air,
                cc, chc,
                float(period_airings.get(it, 0)),
                period_airings.get(it, 0) / total_period,
                float(user_hist_rank.get(u, {}).get(it, 0)),
                float(user_bias.get(u, 0.0)),
                float(item_bias.get(it, 0.0)),
                float(np.dot(ue_l[uidx[u]], ie_l[iidx[it]]))
                if (ue_l is not None and u in uidx and it in iidx) else 0.0,
            ])
        y_groups.append((u, len(cl), len(X_rows)))
    X = np.array(X_rows, dtype=np.float64) if X_rows else np.zeros((0, len(FEATURE_NAMES)))
    return X, y_groups, hist_lookup


# ============================================================ сборка фолда

def build_fold(con, args, hist_alias, hist_sched, hist_where, hist_period_where,
               target_alias, target_sched, target_where, target_period_where,
               ref_date, cat_code, chan_global, chan_id_code, lfm_configs):
    """Готовит один фолд: история + ALS + кандидаты + признаки + метки."""
    log(f"фолд {hist_alias}: строю взаимодействия")
    R.build_interactions(con, hist_alias, hist_sched, hist_where, args.threshold)
    build_rich_agg(con, hist_alias)
    if target_alias:
        R.build_interactions(con, target_alias, target_sched, target_where, args.threshold)
        build_rich_agg(con, target_alias)

    hist_table = f"{hist_alias}_r"
    n_pairs = con.execute(f"SELECT count(*) FROM {hist_table}").fetchone()[0]
    n_users = con.execute(f"SELECT count(DISTINCT user_id) FROM {hist_table}").fetchone()[0]
    n_items = con.execute(f"SELECT count(DISTINCT tv_show_id) FROM {hist_table}").fetchone()[0]
    log(f"  история: пар {n_pairs:,} | юзеров {n_users:,} | программ {n_items:,}")

    # universe = программы, выходящие в целевом периоде
    period_airings = {int(x[0]): int(x[1]) for x in con.execute(f"""
        SELECT tv_show_id, count(*) FROM {target_sched}
        WHERE tv_show_id > 0 {target_period_where} GROUP BY tv_show_id
    """).fetchall()}
    _, _, chan_period = period_meta(con, target_sched, target_period_where)
    universe = set(chan_period.keys())
    log(f"  universe целевого периода: {len(universe)} программ")

    model, items, uidx, iidx = train_als(con, hist_table, args.als_value,
                                          args.factors, args.iterations,
                                          args.regularization, args.seed)
    pop_order = popular_order(con, hist_table, universe)
    log(f"  ALS: items {len(items)}; популярных в universe {len(pop_order)}")

    # глобальная статистика по item
    user_bias, item_bias, ue_f, ie_f = {}, {}, None, None
    bias_variants = {}
    if args.use_biases:
        t_lfm = time.time()
        for cfg in lfm_configs:
            ub, ib, ue, ie = train_lightfm_biases(
                con, hist_table, sorted(uidx.keys()), items, uidx, iidx,
                epochs=cfg["epochs"], components=cfg["components"], seed=args.seed)
            name = f"{cfg['epochs']}ep_{cfg['components']}c"
            bias_variants[name] = (ub, ib, ue, ie)
            if name == f"{args.lfm_epochs}ep_{args.lfm_components}c":
                user_bias, item_bias, ue_f, ie_f = ub, ib, ue, ie
            log(f"  LightFM warp {name}: user_bias [{min(ub.values()):.2f}..{max(ub.values()):.2f}], "
                f"item_bias [{min(ib.values()):.2f}..{max(ib.values()):.2f}]")
        log(f"  LightFM: {len(bias_variants)} конфигов за {time.time()-t_lfm:.1f}s")
    pop_global = {int(x[0]): int(x[1]) for x in con.execute(
        f"SELECT tv_show_id, sum(n_airings) FROM {hist_table} GROUP BY tv_show_id").fetchall()}
    item_users = {int(x[0]): int(x[1]) for x in con.execute(
        f"SELECT tv_show_id, count(DISTINCT user_id) FROM {hist_table} GROUP BY tv_show_id").fetchall()}

    # канал программы: код из целевого периода (где выходит), иначе — из окна истории
    chan_raw = dict(chan_global)
    chan_raw.update(chan_period)
    chan_code = {sid: chan_id_code.get(ch, -1) for sid, ch in chan_raw.items()}

    # --- целевой период: ground truth
    truth, target_users = {}, []
    if target_alias:
        truth = R.make_truth(con, f"{target_alias}_r", k=5)
        target_users = sorted(truth.keys())

    users = target_users if target_users else sorted(uidx.keys())
    cands, flags = gen_candidates(con, hist_table, model, items, uidx, users,
                                  universe, pop_order, args.k_hist, args.k_als,
                                  args.k_pop, args.cap)
    log(f"  кандидатов: всего пар {sum(len(v) for v in cands.values()):,} "
        f"(в среднем {np.mean([len(v) for v in cands.values()]):.1f} на юзера)")

    # признаки
    X, groups, _ = build_features(con, hist_table, users, cands, flags, model, items,
                                  uidx, iidx, universe, cat_code, chan_code, ref_date,
                                  pop_global, item_users, set(pop_order[:args.k_pop]),
                                  period_airings, user_bias, item_bias,
                                  (ue_f, ie_f))
    log(f"  признаки: {X.shape}")

    # метки
    y = np.zeros(X.shape[0], dtype=np.int8)
    if truth:
        ptr = 0
        for u, _, end in groups:
            for it in cands[u]:
                if it in set(truth.get(u, [])):
                    y[ptr] = 1
                ptr += 1
    log(f"  положительных меток: {int(y.sum()):,} / {len(y):,}")
    return dict(X=X, y=y, groups=groups, cands=cands, truth=truth, users=users,
                items=items, uidx=uidx, iidx=iidx, model=model, universe=universe,
                pop_global=pop_global, item_users=item_users, chan_code=chan_code,
                period_airings=period_airings, flags=flags,
                user_bias=user_bias, item_bias=item_bias,
                bias_variants=bias_variants, lfm_emb=(ue_f, ie_f))


def bias_columns(users, cands, ub, ib, ue=None, ie=None, uidx=None, iidx=None):
    """Колонки [user_bias, item_bias, lfm_score] в том же порядке пар, что и в build_features."""
    n = sum(len(cands.get(u, [])) for u in users)
    out = np.zeros((n, 3), dtype=np.float64)
    ptr = 0
    for u in users:
        cl = cands.get(u, [])
        for it in cl:
            out[ptr, 0] = float(ub.get(u, 0.0))
            out[ptr, 1] = float(ib.get(it, 0.0))
            if ue is not None and uidx is not None and u in uidx and it in iidx:
                out[ptr, 2] = float(np.dot(ue[uidx[u]], ie[iidx[it]]))
            ptr += 1
    return out


def feature_cols(X, use_biases=True, which="all"):
    """Индексы используемых колонок.

    which: "all" — все LightFM-фичи, "bias" — только user_bias/item_bias,
           "score" — только lfm_score, "none" — без LightFM.
    """
    if use_biases and which == "all":
        return list(range(X.shape[1]))
    drop = {"all": [], "bias": [FEATURE_NAMES.index("lfm_score")],
            "score": BIAS_IDX, "none": LFM_IDX}[which]
    return [i for i in range(X.shape[1]) if i not in drop]


def fit_ranker(X, y, groups, params, objective="binary", cols=None):
    """Обучает ранкер: бинарный классификатор или lambdarank (с группами по юзерам)."""
    p = dict(params)
    p["objective"] = "binary" if objective == "binary" else "lambdarank"
    if cols is None:
        cols = list(range(X.shape[1]))
    Xs = X[:, cols]
    cat_idx = [i for i, c in enumerate(cols) if c in CAT_IDX]
    if objective == "rank":
        p.setdefault("label_gain", [0, 1])
        model = lgb.LGBMRanker(**p)
        model.fit(Xs, y, group=[g[1] for g in groups], categorical_feature=cat_idx,
                  callbacks=[lgb.log_evaluation(0)])
    else:
        model = lgb.LGBMClassifier(**p)
        model.fit(Xs, y, categorical_feature=cat_idx, callbacks=[lgb.log_evaluation(0)])
    return model


def importance_table(model, cols):
    """Gain по глобальным индексам признаков (0 для колонок, не вошедших в модель)."""
    g = model.booster_.feature_importance(importance_type="gain")
    out = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for pos, i in enumerate(cols):
        out[i] = float(g[pos])
    return out


def predict_topk(model, X, groups, cands, k=5, cols=None):
    """Скор -> Top-k по каждому юзеру."""
    if len(X) == 0:
        return {}
    if cols is None:
        cols = list(range(X.shape[1]))
    Xs = X[:, cols]
    p = model.predict_proba(Xs)[:, 1] if hasattr(model, "predict_proba") else model.predict(Xs)
    out = {}
    ptr = 0
    for u, n, _ in groups:
        if n == 0:
            out[u] = []
            continue
        sc = p[ptr:ptr + n]
        ptr += n
        order = np.argsort(-sc, kind="stable")
        out[u] = [cands[u][i] for i in order[:k]]
    return out


def ap_per_user(preds, truth, k=5):
    """AP@k по каждому юзеру (для парных статистических сравнений)."""
    out = {}
    for u, rel in truth.items():
        rel_set = set(rel)
        hits, acc = 0, 0.0
        for i, item in enumerate(preds.get(u, [])[:k], start=1):
            if item in rel_set:
                hits += 1
                acc += hits / i
        out[u] = acc / min(k, len(rel_set)) if rel_set else 0.0
    return out


def paired_delta(preds_a, preds_b, truth, k=5):
    """Парная разница AP (b - a) по юзерам с полным GT: среднее и стандартная ошибка."""
    a, b = ap_per_user(preds_a, truth, k), ap_per_user(preds_b, truth, k)
    users = [u for u, rel in truth.items() if len(rel) >= k]
    d = np.array([b[u] - a[u] for u in users])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, len(d)


def map_report(preds, truth, label):
    strict, n_u, n_skip = R.map_at_k(preds, truth, k=5, require_full=True)
    soft, _, _ = R.map_at_k(preds, truth, k=5, require_full=False)
    rec = R.recall_at_k(preds, truth, k=5)
    cov = len({i for v in preds.values() for i in v})
    log(f"{label}: MAP@5 строгий {strict:.4f} | мягкий {soft:.4f} | Recall@5 {rec:.4f} | "
        f"покрытие {cov}")
    return dict(strict=strict, soft=soft, recall=rec, coverage=cov, n_users=n_u)


# ============================================================ главный сценарий

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--mode", default="validation", choices=["validation", "submit", "both"])
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--als-value", default="airings", choices=["airings", "screen_time"])
    ap.add_argument("--factors", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--regularization", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-hist", type=int, default=50)
    ap.add_argument("--k-als", type=int, default=100)
    ap.add_argument("--k-pop", type=int, default=50)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=31)
    ap.add_argument("--min-child-samples", type=int, default=50)
    ap.add_argument("--template", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-features", default=None, help="npz с парами и признаками")
    ap.add_argument("--tune", action="store_true", help="вложенный тюнинг LightGBM (sub->inner)")
    ap.add_argument("--use-mid", action="store_true", default=True,
                    help="добавить фолд mid (15w->26-30) в обучение финального ранкера")
    ap.add_argument("--no-mid", dest="use_mid", action="store_false")
    ap.add_argument("--objective", default="binary", choices=["binary", "rank"])
    ap.add_argument("--use-biases", action="store_true", default=True,
                    help="добавить user_bias/item_bias из LightFM")
    ap.add_argument("--no-biases", dest="use_biases", action="store_false")
    ap.add_argument("--lfm-epochs", type=int, default=30)
    ap.add_argument("--lfm-components", type=int, default=20)
    ap.add_argument("--final-variant", default="auto",
                    choices=["auto", "none", "bias", "score", "all"],
                    help="LightFM-фичи в финальном сабмите (auto = лучший на validation)")
    ap.add_argument("--bias-seeds", action="store_true",
                    help="проверить устойчивость к сиду LightGBM (3 сида)")
    ap.add_argument("--lfm-sweep", action="store_true",
                    help="перебрать конфиги LightFM (epochs/components) на валидации")
    args = ap.parse_args()

    data_dir = args.data_dir
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='8GB';")

    split = R.load_data(con, data_dir)
    R.build_metadata(con)
    log(f"SPLIT (неделя 21) = {split}")

    # внутренний сплит: недели 16–20 (5 недель после старта 11)
    mn = con.execute("SELECT min(program_start) FROM sched_train").fetchone()[0]
    inner_split = con.execute(
        f"SELECT min(program_start) FROM sched_train WHERE program_start >= TIMESTAMP '{mn}' + INTERVAL 35 DAY"
    ).fetchone()[0]
    # доп. границы: неделя 14 (для вложенного тюнинга) и неделя 26 (для финальной модели)
    sub_split = con.execute(
        f"SELECT min(program_start) FROM sched_train WHERE program_start >= TIMESTAMP '{mn}' + INTERVAL 21 DAY"
    ).fetchone()[0]
    mid_split = con.execute(
        f"SELECT min(program_start) FROM sched_valid WHERE program_start >= TIMESTAMP '{split}' + INTERVAL 35 DAY"
    ).fetchone()[0]
    test_start = con.execute("SELECT min(program_start) FROM sched_test").fetchone()[0]
    log(f"внутренний сплит (неделя 16) = {inner_split}; sub (неделя 14) = {sub_split}; "
        f"mid (неделя 26) = {mid_split}; старт теста = {test_start}")

    con.execute("CREATE OR REPLACE TABLE sched_all AS SELECT * FROM sched_train UNION ALL SELECT * FROM sched_valid")
    lfm_configs = [{"epochs": args.lfm_epochs, "components": args.lfm_components}]
    if args.lfm_sweep:
        lfm_configs += [{"epochs": e, "components": c}
                        for e, c in [(100, 20), (300, 20), (100, 10), (300, 1)]]
    cat_code, cat_names = global_category_map(con)
    chan_global = {}
    for alias in ["sched_train", "sched_valid", "sched_test"]:
        _, _, ch = period_meta(con, alias)
        for sid, c in ch.items():
            chan_global.setdefault(sid, c)
    chan_id_code = {ch: i + 1 for i, ch in enumerate(sorted(set(chan_global.values())))}
    log(f"категорий: {len(cat_names)}, программ с категорией: {len(cat_code)}, "
        f"каналов: {len(chan_id_code)}")

    # ------------------------------------------------------------------ folds
    results = {}
    params = dict(objective="binary", n_estimators=args.n_estimators,
                  learning_rate=args.learning_rate, num_leaves=args.num_leaves,
                  min_child_samples=args.min_child_samples, subsample=0.9,
                  subsample_freq=1, colsample_bytree=0.9, random_state=args.seed,
                  n_jobs=8, verbose=-1)

    if args.mode in ("validation", "both"):
        log("=" * 90)
        log("FOLD inner: история 11–15 -> метки 16–20 (обучение ранкера)")
        inner = build_fold(
            con, args,
            hist_alias="in", hist_sched="sched_train",
            hist_where=f"l.session_start < TIMESTAMP '{inner_split}'",
            hist_period_where="",
            target_alias="igt", target_sched="sched_train",
            target_where=f"l.session_start >= TIMESTAMP '{inner_split}'",
            target_period_where=f"AND program_start >= TIMESTAMP '{inner_split}'",
            ref_date=inner_split, cat_code=cat_code, chan_global=chan_global,
            chan_id_code=chan_id_code, lfm_configs=lfm_configs)

        # ---------- вложенный тюнинг: обучение на sub (11–13->14–15), оценка на inner
        if args.tune:
            log("=" * 90)
            log("ТЮНИНГ: FOLD sub (история 11–13 -> метки 14–15), оценка на inner (11–15->16–20)")
            sub = build_fold(
                con, args,
                hist_alias="sb", hist_sched="sched_train",
                hist_where=f"l.session_start < TIMESTAMP '{sub_split}'",
                hist_period_where="",
                target_alias="sbt", target_sched="sched_train",
                target_where=f"l.session_start >= TIMESTAMP '{sub_split}'",
                target_period_where=f"AND program_start >= TIMESTAMP '{sub_split}'",
                ref_date=sub_split, cat_code=cat_code, chan_global=chan_global,
                chan_id_code=chan_id_code, lfm_configs=lfm_configs)
            grid = [
                dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50),
                dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50),
                dict(n_estimators=200, learning_rate=0.05, num_leaves=63, min_child_samples=50),
                dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50),
                dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=20),
                dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100),
                dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20),
            ]
            best, best_score = None, -1
            for g in grid:
                p = dict(params)
                p.update(g)
                m = fit_ranker(sub["X"], sub["y"], sub["groups"], p, args.objective)
                pr = predict_topk(m, inner["X"], inner["groups"], inner["cands"], k=5)
                sc, _, _ = R.map_at_k(pr, inner["truth"], k=5, require_full=True)
                log(f"  конфиг {g} -> MAP@5(inner) {sc:.4f}")
                if sc > best_score:
                    best, best_score = g, sc
            params.update(best)
            log(f"лучший конфиг: {best} (MAP@5 на inner = {best_score:.4f})")
            results["sub"] = sub

        log("=" * 90)
        log("FOLD valid: история 11–20 -> метки 21–30 (честная оценка)")
        valid = build_fold(
            con, args,
            hist_alias="tr", hist_sched="sched_train",
            hist_where=f"l.session_start < TIMESTAMP '{split}'",
            hist_period_where="",
            target_alias="va", target_sched="sched_valid",
            target_where=f"l.session_start >= TIMESTAMP '{split}'",
            target_period_where="",
            ref_date=split, cat_code=cat_code, chan_global=chan_global,
            chan_id_code=chan_id_code, lfm_configs=lfm_configs)

        # ---------- baseline ALS (тот же кандидатный universe, что и у two-layer)
        log("=" * 90)
        log("BASELINE: ALS top-5 на validation (для сравнения)")
        model_v = valid["model"]
        users_v = valid["users"]
        items_v = valid["items"]
        uidx_v, iidx_v = valid["uidx"], valid["iidx"]
        uf = np.asarray(model_v.user_factors)
        itf = np.asarray(model_v.item_factors)
        items_arr = np.asarray(items_v, dtype=np.int64)
        mask = np.array([it in valid["universe"] for it in items_v], dtype=bool)
        als_preds = {}
        for u in users_v:
            if u not in uidx_v:
                als_preds[u] = []
                continue
            s = uf[uidx_v[u]] @ itf.T
            s[~mask] = -np.inf
            idx = np.argpartition(-s, 9)[:9]
            idx = idx[np.isfinite(s[idx])]
            order = np.lexsort((items_arr[idx], -s[idx]))
            als_preds[u] = [int(items_arr[idx[i]]) for i in order[:5]]
        base = map_report(als_preds, valid["truth"], "ALS baseline (top-5, universe=эфир 21–30)")

        # ---------- потолок кандидатов
        rec_cand = []
        for u, gt in valid["truth"].items():
            cs = set(valid["cands"].get(u, []))
            rec_cand.append(len(cs & set(gt)) / max(len(gt), 1))
        log(f"потолок кандидатов: recall@200 кандидатов = {np.mean(rec_cand):.4f}")
        cov_src = {"история": 0, "ALS": 0, "популярные": 0, "не покрыт": 0}
        n_gt = 0
        for u, gt in valid["truth"].items():
            fl = valid["flags"].get(u, {})
            for it in gt:
                n_gt += 1
                f = fl.get(it)
                if f is None:
                    cov_src["не покрыт"] += 1
                else:
                    if f[0]:
                        cov_src["история"] += 1
                    elif f[1]:
                        cov_src["ALS"] += 1
                    else:
                        cov_src["популярные"] += 1
        log("покрытие GT-элементов источниками: " +
            ", ".join(f"{k} {v} ({v/n_gt:.1%})" for k, v in cov_src.items()) + f" | всего {n_gt}")

        # ---------- фолд mid: история 11–25 -> метки 26–30 (для проверки переноса
        #            ранкера с короткой истории на длинную + обучение финальной модели)
        mid = None
        if args.use_mid:
            log("=" * 90)
            log("FOLD mid: история 11–25 -> метки 26–30")
            mid = build_fold(
                con, args,
                hist_alias="md", hist_sched="sched_all",
                hist_where=f"l.session_start < TIMESTAMP '{mid_split}'",
                hist_period_where="",
                target_alias="mdt", target_sched="sched_valid",
                target_where=f"l.session_start >= TIMESTAMP '{mid_split}'",
                target_period_where=f"AND program_start >= TIMESTAMP '{mid_split}'",
                ref_date=mid_split, cat_code=cat_code, chan_global=chan_global,
                chan_id_code=chan_id_code, lfm_configs=lfm_configs)
            results["mid"] = mid

        # ---------- обучение ранкера на inner, оценка на valid
        log("=" * 90)
        log("TWO-LAYER: обучение на inner (11–15->16–20), оценка на valid (11–20->21–30)")

        # (a) без LightFM  (b) user_bias+item_bias (как в ТЗ)  (c) lfm_score  (d) все три
        def eval_variant(which, label):
            cols = feature_cols(valid["X"], use_biases=True, which=which)
            m = fit_ranker(inner["X"], inner["y"], inner["groups"], params,
                           args.objective, cols)
            pr = predict_topk(m, valid["X"], valid["groups"], valid["cands"], 5, cols)
            return cols, pr, map_report(pr, valid["truth"], label), m

        cols_nb, preds_nb, res_nb, clf_nb = eval_variant("none", "(a) TWO-LAYER БЕЗ LightFM")
        cols_b, preds_b, res_b, clf_b = eval_variant("bias", "(b) TWO-LAYER + user_bias/item_bias")
        cols_s, preds_s, res_s, clf_s = eval_variant("score", "(c) TWO-LAYER + lfm_score")
        cols_a, preds_a, res_a, clf_a = eval_variant("all", "(d) TWO-LAYER + все 3 LightFM-фичи")

        d = res_b["strict"] - res_nb["strict"]
        log("=" * 90)
        log(f"СРАВНЕНИЕ: биасы LightFM дают {d:+.4f} MAP@5 "
            f"({'прирост' if d > 0 else 'НЕТ прироста'})")
        variants = {"none": res_nb, "bias": res_b, "score": res_s, "all": res_a}
        best_which = max(variants, key=lambda k: variants[k]["strict"])
        log(f"лучший вариант: {best_which} ({variants[best_which]['strict']:.4f}); "
            f"дельта к варианту без LightFM: "
            f"{variants[best_which]['strict'] - res_nb['strict']:+.4f}")
        # ---------- парная статистика: отличается ли разница от шума?
        for lbl, pa, pb in [("биасы", preds_nb, preds_b), ("lfm_score", preds_nb, preds_s),
                            ("все 3", preds_nb, preds_a)]:
            m, se, n = paired_delta(pa, pb, valid["truth"])
            log(f"парное сравнение (без LightFM vs {lbl}): ΔAP = {m:+.4f} ± {se:.4f} "
                f"(SE, n={n}) -> {'значимо' if abs(m) > 2 * se else 'в пределах шума'}")

        # ---------- устойчивость к сиду LightGBM
        if args.bias_seeds:
            log("=" * 90)
            log("УСТОЙЧИВОСТЬ: 3 сида LightGBM для каждого варианта")
            seed_rows = {}
            for which in ["none", "bias", "score", "all"]:
                cols = feature_cols(valid["X"], use_biases=True, which=which)
                scs = []
                for sd in [42, 7, 2024]:
                    pp = dict(params)
                    pp["random_state"] = sd
                    m = fit_ranker(inner["X"], inner["y"], inner["groups"], pp,
                                   args.objective, cols)
                    pr = predict_topk(m, valid["X"], valid["groups"], valid["cands"], 5, cols)
                    scs.append(R.map_at_k(pr, valid["truth"], k=5, require_full=True)[0])
                seed_rows[which] = scs
                log(f"   {which:<6}: " + ", ".join(f"{x:.4f}" for x in scs) +
                    f"  среднее {np.mean(scs):.4f} (разброс {np.std(scs):.4f})")
            for which in ["bias", "score", "all"]:
                dm = np.mean(seed_rows[which]) - np.mean(seed_rows["none"])
                log(f"   дельта '{which}' - 'none' по 3 сидам: {dm:+.4f}")

        results_cmp = dict(no_bias=res_nb, with_bias=res_b, score=res_s, all=res_a,
                           delta=d, best_which=best_which,
                           cols={"none": cols_nb, "bias": cols_b, "score": cols_s, "all": cols_a})

        # ---------- перебор конфигов LightFM: возможно, биасы недообучены
        if args.lfm_sweep and inner.get("bias_variants") and valid.get("bias_variants"):
            log("=" * 90)
            log("ПЕРЕБОР КОНФИГОВ LIGHTFM (ранкер обучается на inner, оценивается на valid)")
            sweep_rows = []
            for name in inner["bias_variants"]:
                X_in = inner["X"].copy()
                ub, ib, ue, ie = inner["bias_variants"][name]
                ub2, ib2, ue2, ie2 = valid["bias_variants"][name]
                X_in[:, -3:] = bias_columns(inner["users"], inner["cands"], ub, ib, ue, ie,
                                            inner["uidx"], inner["iidx"])
                X_va = valid["X"].copy()
                X_va[:, -3:] = bias_columns(valid["users"], valid["cands"], ub2, ib2, ue2, ie2,
                                            valid["uidx"], valid["iidx"])
                cols = list(range(X_in.shape[1]))
                m = fit_ranker(X_in, inner["y"], inner["groups"], params,
                               args.objective, cols)
                pr = predict_topk(m, X_va, valid["groups"], valid["cands"], 5, cols)
                r = map_report(pr, valid["truth"], f"  LightFM {name}")
                sweep_rows.append((name, r["strict"], r["soft"], r["recall"]))
            log("сводка по конфигам LightFM:")
            for name, a, b, c in sweep_rows:
                log(f"   {name:<12} MAP@5 {a:.4f} | мягкий {b:.4f} | Recall {c:.4f} "
                    f"| дельта к базе {a - res_nb['strict']:+.4f}")
            best_name, best_sc = max(sweep_rows, key=lambda r: r[1])
            log(f"лучший конфиг LightFM: {best_name} ({best_sc:.4f}); "
                f"база без биасов {res_nb['strict']:.4f}")
            if best_sc > res_nb["strict"]:
                results_cmp["best_cfg"] = best_name
                results_cmp["delta"] = best_sc - res_nb["strict"]
                log(f"применяем конфиг {best_name} в финальной модели")
            else:
                results_cmp["best_cfg"] = None

        # основная модель для дальнейших замеров — лучший вариант по validation
        clfs = {"none": clf_nb, "bias": clf_b, "score": clf_s, "all": clf_a}
        preds_all = {"none": preds_nb, "bias": preds_b, "score": preds_s, "all": preds_a}
        clf, preds, two = clfs[best_which], preds_all[best_which], variants[best_which]
        log(f"для дальнейших замеров используется вариант '{best_which}'")

        # ---------- для сравнения: ранкер, обученный на самой validation (leaky)
        clf_leaky = fit_ranker(valid["X"], valid["y"], valid["groups"], params, args.objective)
        preds_leaky = predict_topk(clf_leaky, valid["X"], valid["groups"], valid["cands"], k=5)
        leaky = map_report(preds_leaky, valid["truth"],
                           "TWO-LAYER (оптимистично: обучение на самой validation)")

        # ---------- важность признаков (gain), модель с биасами
        imp = importance_table(clf_b, cols_b)
        imp_nb = importance_table(clf_nb, cols_nb)
        order = np.argsort(-imp)
        log("важность признаков (модель С биасами, обучена на inner; gain):")
        for i in order:
            mark = "  <-- LightFM" if i in BIAS_IDX else ""
            ref = f"{imp_nb[i]:>12.0f}" if i in cols_nb else " " * 12
            log(f"   {FEATURE_NAMES[i]:<22} {imp[i]:>12.0f}   (без биасов: {ref}){mark}")
        log(f"вклад LightFM-фич: user_bias {imp[BIAS_IDX[0]]:.0f}, "
            f"item_bias {imp[BIAS_IDX[1]]:.0f}, lfm_score {imp[LFM_IDX[2]]:.0f} "
            f"(сумма gain всех признаков {imp.sum():.0f})")

        # ---------- что улучшилось/ухудшилось по юзерам
        gtb = {u: set(valid["truth"][u]) for u in valid["truth"]}
        gain_u = []
        for u, gt in gtb.items():
            if len(gt) < 5:
                continue
            ap_a = R.map_at_k({u: als_preds.get(u, [])}, {u: list(gt)}, k=5, require_full=True)[0]
            ap_b = R.map_at_k({u: preds.get(u, [])}, {u: list(gt)}, k=5, require_full=True)[0]
            gain_u.append(ap_b - ap_a)
        gain_u = np.array(gain_u)
        log(f"пер-юзерный выигрыш two-layer vs ALS: средний {gain_u.mean():+.4f}, "
            f"лучше {np.mean(gain_u>0):.1%}, хуже {np.mean(gain_u<0):.1%}, "
            f"худший {gain_u.min():+.3f}")

        # ---------- ПРОВЕРКА ПЕРЕНОСА: ранкер, обученный на короткой истории (5 недель),
        #            применённый к длинной (15 недель). Периоды меток не пересекаются
        #            (16–20 против 26–30) -> честная оценка.
        if mid is not None:
            log("=" * 90)
            log("ПРОВЕРКА ПЕРЕНОСА: обучение на inner (5w) -> применение к mid (15w)")
            clf_shift = fit_ranker(inner["X"], inner["y"], inner["groups"], params, args.objective)
            pr_shift = predict_topk(clf_shift, mid["X"], mid["groups"], mid["cands"], k=5)
            map_report(pr_shift, mid["truth"], "TWO-LAYER на mid (обучен на inner, 5w->15w)")
            # ALS-базовая линия на том же фолде
            uf_m = np.asarray(mid["model"].user_factors)
            itf_m = np.asarray(mid["model"].item_factors)
            itm_m = np.asarray(mid["items"], dtype=np.int64)
            msk_m = np.array([it in mid["universe"] for it in mid["items"]], dtype=bool)
            als_m = {}
            for u in mid["users"]:
                if u not in mid["uidx"]:
                    als_m[u] = []
                    continue
                sc = uf_m[mid["uidx"][u]] @ itf_m.T
                sc[~msk_m] = -np.inf
                idx = np.argpartition(-sc, 9)[:9]
                idx = idx[np.isfinite(sc[idx])]
                order = np.lexsort((itm_m[idx], -sc[idx]))
                als_m[u] = [int(itm_m[idx[i]]) for i in order[:5]]
            map_report(als_m, mid["truth"], "ALS baseline на mid")

        results.update(dict(base=base, two=two, leaky=leaky, imp=imp,
                            inner=inner, valid=valid, cmp=results_cmp,
                            cols_b=cols_b, cols_nb=cols_nb))

    # ------------------------------------------------------------------ сабмит
    if args.mode in ("submit", "both"):
        log("=" * 90)
        log("САБМИТ: история 11–30, кандидаты = эфир 31–42")
        full = build_fold(
            con, args,
            hist_alias="fu", hist_sched="sched_train",
            hist_where=f"l.session_start < TIMESTAMP '{split}'",
            hist_period_where="",
            target_alias="fv", target_sched="sched_valid",
            target_where=f"l.session_start >= TIMESTAMP '{split}'",
            target_period_where="",
            ref_date=split, cat_code=cat_code, chan_global=chan_global,
            chan_id_code=chan_id_code, lfm_configs=lfm_configs)
        # полная история 11–30 = tr + va
        con.execute("""
          CREATE OR REPLACE TABLE full_r AS
          SELECT user_id, tv_show_id, sum(n_airings)::BIGINT n_airings,
                 max(max_screen_time) max_screen_time, sum(sum_screen_time) sum_screen_time,
                 sum(total_overlap) total_overlap, max(last_watch) last_watch,
                 mode(channel_id) channel_id,
                 any_value(tv_show_title) tv_show_title, any_value(tv_show_category) tv_show_category,
                 any_value(tv_show_genre_1) tv_show_genre_1, any_value(tv_show_genre_2) tv_show_genre_2,
                 any_value(tv_show_genre_3) tv_show_genre_3
          FROM (SELECT * FROM fu_r UNION ALL SELECT * FROM fv_r) GROUP BY user_id, tv_show_id
        """)
        log(f"полная история 11–30: пар {con.execute('SELECT count(*) FROM full_r').fetchone()[0]:,}")

        # кандидаты для тестовых юзеров: universe = эфир 31–42
        template = args.template or os.path.join(data_dir, "submission.csv")
        out_path = args.out or os.path.join(data_dir, "submission_with_biases.csv")
        test_users = []
        with open(template, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                if row and row[0].strip():
                    test_users.append(row[0].strip())
        log(f"шаблон: {len(test_users)} тестовых юзеров")

        _, _, chan_test = period_meta(con, "sched_test")
        universe_test = set(chan_test.keys())
        model_f, items_f, uidx_f, iidx_f = train_als(con, "full_r", args.als_value,
                                                      args.factors, args.iterations,
                                                      args.regularization, args.seed)
        pop_order_f = popular_order(con, "full_r", universe_test)
        tu = [int(u) for u in test_users]
        cands_f, flags_f = gen_candidates(con, "full_r", model_f, items_f, uidx_f, tu,
                                          universe_test, pop_order_f, args.k_hist,
                                          args.k_als, args.k_pop, args.cap)
        log(f"кандидатов: {sum(len(v) for v in cands_f.values()):,} пар")
        pop_global_f = {int(x[0]): int(x[1]) for x in con.execute(
            "SELECT tv_show_id, sum(n_airings) FROM full_r GROUP BY tv_show_id").fetchall()}
        item_users_f = {int(x[0]): int(x[1]) for x in con.execute(
            "SELECT tv_show_id, count(DISTINCT user_id) FROM full_r GROUP BY tv_show_id").fetchall()}
        chan_raw_f = dict(chan_global)
        chan_raw_f.update(chan_test)
        chan_code_f = {sid: chan_id_code.get(ch, 0) for sid, ch in chan_raw_f.items()}
        period_airings_f = {int(x[0]): int(x[1]) for x in con.execute("""
            SELECT tv_show_id, count(*) FROM sched_test WHERE tv_show_id > 0
            GROUP BY tv_show_id""").fetchall()}
        user_bias_f, item_bias_f, ue_f, ie_f = {}, {}, None, None
        if args.use_biases:
            t_lfm = time.time()
            users_f = sorted(uidx_f.keys())
            user_bias_f, item_bias_f, ue_f, ie_f = train_lightfm_biases(
                con, "full_r", users_f, items_f, uidx_f, iidx_f,
                epochs=args.lfm_epochs, components=args.lfm_components, seed=args.seed)
            log(f"LightFM на полной истории 11–30: {len(user_bias_f)} юзеров, "
                f"{len(item_bias_f)} программ ({time.time()-t_lfm:.1f}s)")
        X_f, groups_f, _ = build_features(con, "full_r", tu, cands_f, flags_f, model_f,
                                          items_f, uidx_f, iidx_f, universe_test, cat_code,
                                          chan_code_f, test_start, pop_global_f, item_users_f,
                                          set(pop_order_f[:args.k_pop]), period_airings_f,
                                          user_bias_f, item_bias_f, (ue_f, ie_f))
        log(f"признаки для теста: {X_f.shape}")

        # ранкер: обучаем на inner + valid + mid
        # (легитимно: все метки из недель 16–30, т.е. строго до тестового периода 31–42)
        X_tr, y_tr, names = [], [], []
        for key, lbl in [("inner", "inner(5w->16-20)"), ("valid", "valid(10w->21-30)")]:
            if key in results:
                X_tr.append(results[key]["X"])
                y_tr.append(results[key]["y"])
                names.append(lbl)
        if "mid" in results:
            X_tr.append(results["mid"]["X"])
            y_tr.append(results["mid"]["y"])
            names.append("mid(15w->26-30)")
        if not X_tr:
            X_tr, y_tr = [X_f], [np.zeros(len(X_f), dtype=np.int8)]
        X_tr = np.vstack(X_tr)
        y_tr = np.concatenate(y_tr)
        groups_tr = []
        for key, lbl in [("inner", "inner"), ("valid", "valid")]:
            if key in results:
                groups_tr.extend(results[key]["groups"])
        if args.use_mid and "mid" in results:
            groups_tr.extend(results["mid"]["groups"])
        which = args.final_variant
        if which == "auto":
            which = results.get("cmp", {}).get("best_which", "none")
        if which not in ("none", "bias", "score", "all"):
            which = "none"
        cols_final = results.get("cmp", {}).get("cols", {}).get(which)
        if cols_final is None:
            cols_final = feature_cols(X_f, use_biases=args.use_biases, which=which)
        log(f"финальный ранкер: вариант LightFM '{which}' "
            f"({len(cols_final)}/{X_f.shape[1]} признаков)")
        clf_final = fit_ranker(X_tr, y_tr, groups_tr, params, args.objective, cols_final)
        log(f"ранкер обучен на {X_tr.shape[0]:,} парах ({' + '.join(names)})")
        imp_f = importance_table(clf_final, cols_final)
        log("важность признаков финальной модели (gain):")
        for i in np.argsort(-imp_f):
            mark = "  <-- LightFM" if i in BIAS_IDX else ""
            log(f"   {FEATURE_NAMES[i]:<22} {imp_f[i]:>12.0f}{mark}")

        top = predict_topk(clf_final, X_f, groups_f, cands_f, 5, cols_final)
        pop_top = [it for it in pop_order_f[:5] if it in universe_test]
        rows_out = []
        for u in test_users:
            uid = int(u)
            rec = list(top.get(uid, []))
            for it in pop_top:
                if len(rec) == 5:
                    break
                if it not in rec:
                    rec.append(it)
            rec = rec[:5]
            rows_out.append([u, " ".join(str(i) for i in rec)])
        tmp = out_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["user_id", "tv_show_id"])
            w.writerows(rows_out)
        os.replace(tmp, out_path)
        log(f"сохранено {len(rows_out)} строк -> {out_path}")
        ids = [int(x) for _, s in rows_out for x in s.split(" ")]
        log(f"проверка: уникальных id {len(set(ids))}, нулей {ids.count(0)}, "
            f"вне эфира 31–42 {sum(1 for i in set(ids) if i not in universe_test)}, "
            f"дублей в строке {sum(1 for _, s2 in rows_out if len(set(s2.split(' '))) != 5)}")

        # ---------- отчёт: юзер с историей и cold-start юзер
        title_m, cat_m, _ = period_meta(con, "sched_test")
        hist_users = {int(x[0]) for x in con.execute("SELECT DISTINCT user_id FROM full_r").fetchall()}
        with_hist = [u for u in test_users if int(u) in hist_users]
        without = [u for u in test_users if int(u) not in hist_users]
        def report(uid, title):
            rec = dict(rows_out)[uid].split(" ")
            print(f"\n{title}: user_id = {uid}")
            if int(uid) in hist_users:
                h = con.execute(f"""SELECT count(*), sum(n_airings) FROM full_r
                                   WHERE user_id = {int(uid)}""").fetchone()
                print(f"  история 11–30: {h[0]} программ, {h[1]} эфиров")
            else:
                print("  история 11–30: ПУСТО -> cold start (популярный Топ-5)")
            for i, it in enumerate(rec, 1):
                it = int(it)
                print(f"  {i}. tv_show_id={it:<10} {str(title_m.get(it,'?'))[:52]:<52} "
                      f"категория: {cat_m.get(it,'?')}")
        if with_hist:
            report(with_hist[0], "Пользователь с историей")
        if without:
            report(without[0], "Пользователь без истории (cold start)")

    con.close()
    log("готово")


if __name__ == "__main__":
    main()
