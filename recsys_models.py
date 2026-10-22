#!/usr/bin/env python3
"""
Три подхода к Top-5 рекомендациям на данных SweetTV (Kaggle: sweettv-tv-program-recommender).

Разбиение по времени (границы берутся из самих файлов расписания, не захардкожены):
  TRAIN      — недели 11–20: export_arh_11-20-final.csv, сессии с session_start < SPLIT
  VALIDATION — недели 21–30: export_arh_21-30-final.csv, сессии с session_start >= SPLIT
  TEST       — недели 31–42: export_arh_31-42-final.csv (только метаданные и кандидаты)

Модели:
  1) Popularity — глобальный Top-5 по частоте в train.
  2) ALS        — implicit.als.AlternatingLeastSquares на матрице user x item.
  3) CBF        — профиль юзера = распределение по tv_show_category + жанрам,
                  кандидаты = программы, вышедшие в эфир в недели 21–30.

Метрика: MAP@5 (см. map_at_k).

Гиперпараметры ALS и вес popularity-приора в CBF подбираются на ВНУТРЕННЕМ сплите
(недели 11–15 train / 16–20 validation), чтобы не подбирать их на отчётном validation.

Запуск:
  python3 recsys_models.py
  python3 recsys_models.py --selftest          # юнит-тесты MAP@5
  python3 recsys_models.py --no-sweep          # без подбора гиперпараметров
"""

import argparse
import os
import time
from collections import defaultdict

# implicit просит однопоточный BLAS (иначе бывают жёсткие тормоза и краши)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import duckdb
import numpy as np
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

TRAIN_SCHED = "export_arh_11-20-final.csv"
VALID_SCHED = "export_arh_21-30-final.csv"
TEST_SCHED = "export_arh_31-42-final.csv"

SCHED_COLS = (
    "{channel_id:'VARCHAR',channel_title:'VARCHAR',start_time:'VARCHAR',duration:'VARCHAR',"
    "tv_show_title:'VARCHAR',tv_show_id:'VARCHAR',tv_show_category:'VARCHAR',"
    "tv_show_genre_1:'VARCHAR',tv_show_genre_2:'VARCHAR',tv_show_genre_3:'VARCHAR',"
    "year_of_production:'VARCHAR',director:'VARCHAR','actors':'VARCHAR'}"
)
LOG_COLS = ("{user_id:'VARCHAR','vsetv_id':'VARCHAR','start_time':'VARCHAR',"
            "'stop_time':'VARCHAR','duraton':'VARCHAR'}")


# ============================================================ загрузка и подготовка
def load_data(con, data_dir):
    con.execute(f"""
      CREATE OR REPLACE TABLE logs AS
      SELECT try_cast(user_id AS HUGEINT)   AS user_id,
             try_cast(vsetv_id AS BIGINT)   AS channel_id,
             strptime(start_time,'%Y-%m-%d %H:%M:%S') AS session_start,
             strptime(stop_time,  '%Y-%m-%d %H:%M:%S') AS session_stop
      FROM read_csv('{os.path.join(data_dir, 'dataset11-30.csv')}', header=true, quote='"',
                    columns={LOG_COLS}, sample_size=-1)
      WHERE strptime(stop_time,'%Y-%m-%d %H:%M:%S') > strptime(start_time,'%Y-%m-%d %H:%M:%S')
    """)

    for alias, fname in [("sched_train", TRAIN_SCHED), ("sched_valid", VALID_SCHED),
                         ("sched_test", TEST_SCHED)]:
        con.execute(f"""
          CREATE OR REPLACE TABLE {alias} AS
          SELECT try_cast(channel_id AS BIGINT) AS channel_id,
                 strptime(start_time,'%d.%m.%Y %H:%M:%S') AS program_start,
                 try_cast(duration AS BIGINT) AS duration,
                 strptime(start_time,'%d.%m.%Y %H:%M:%S')
                     + try_cast(duration AS BIGINT) * INTERVAL 1 SECOND AS program_end,
                 try_cast(tv_show_id AS BIGINT) AS tv_show_id,
                 tv_show_title, tv_show_category,
                 tv_show_genre_1, tv_show_genre_2, tv_show_genre_3
          FROM read_csv('{os.path.join(data_dir, fname)}', header=true, quote='"',
                        columns={SCHED_COLS}, sample_size=-1)
          WHERE try_cast(duration AS BIGINT) > 0
        """)
    split = con.execute("SELECT min(program_start) FROM sched_valid").fetchone()[0]
    return split


def build_interactions(con, alias, sched_alias, where, threshold=0.8):
    """Сессия x программа -> пересечение -> screen_time >= threshold -> агрегат (user, show)."""
    con.execute(f"""
      CREATE OR REPLACE TABLE {alias}_int AS
      SELECT l.user_id, p.tv_show_id, p.tv_show_title, p.tv_show_category,
             p.tv_show_genre_1, p.tv_show_genre_2, p.tv_show_genre_3,
             p.channel_id,
             l.session_start, l.session_stop, p.program_start, p.duration,
             GREATEST(0, date_diff('second',
                        GREATEST(l.session_start, p.program_start),
                        LEAST(l.session_stop,  p.program_end))) AS overlap_seconds,
             GREATEST(0, date_diff('second',
                        GREATEST(l.session_start, p.program_start),
                        LEAST(l.session_stop,  p.program_end)))::DOUBLE
                / p.duration::DOUBLE AS screen_time
      FROM logs l
      JOIN {sched_alias} p
        ON l.channel_id = p.channel_id
       AND l.session_start < p.program_end     -- сессия началась до конца программы
       AND l.session_stop  > p.program_start   -- сессия закончилась после начала программы
      WHERE {where} AND p.tv_show_id > 0 AND p.duration > 0
    """)
    con.execute(f"DELETE FROM {alias}_int WHERE screen_time < {threshold}")
    con.execute(f"""
      CREATE OR REPLACE TABLE {alias}_agg AS
      SELECT user_id, tv_show_id,
             count(*) AS n_airings,
             max(screen_time) AS max_screen_time,
             sum(overlap_seconds) AS total_overlap,
             any_value(tv_show_title) AS tv_show_title,
             any_value(tv_show_category) AS tv_show_category,
             any_value(tv_show_genre_1) AS tv_show_genre_1,
             any_value(tv_show_genre_2) AS tv_show_genre_2,
             any_value(tv_show_genre_3) AS tv_show_genre_3
      FROM {alias}_int
      GROUP BY user_id, tv_show_id
    """)


def build_metadata(con):
    """Справочник метаданных из ВСЕХ файлов расписания.

    Нужен, потому что ~90% строк расписания имеют tv_show_id = 0 и/или пустую категорию:
    недостающие поля добираем по названию программы.
    """
    union = "(SELECT * FROM sched_train UNION ALL SELECT * FROM sched_valid UNION ALL SELECT * FROM sched_test)"
    con.execute(f"""
      CREATE OR REPLACE TABLE meta_by_id AS
      SELECT tv_show_id,
             arg_max(tv_show_category, cnt) AS tv_show_category,
             arg_max(tv_show_genre_1, cnt) AS tv_show_genre_1,
             arg_max(tv_show_genre_2, cnt) AS tv_show_genre_2,
             arg_max(tv_show_genre_3, cnt) AS tv_show_genre_3
      FROM (
        SELECT tv_show_id, tv_show_category, tv_show_genre_1, tv_show_genre_2, tv_show_genre_3,
               count(*) OVER (PARTITION BY tv_show_id, tv_show_category, tv_show_genre_1,
                                              tv_show_genre_2, tv_show_genre_3) AS cnt
        FROM {union} WHERE tv_show_id > 0
      ) GROUP BY tv_show_id
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE meta_by_title AS
      SELECT tv_show_title,
             arg_max(tv_show_category, cnt) AS tv_show_category,
             arg_max(tv_show_genre_1, cnt) AS tv_show_genre_1,
             arg_max(tv_show_genre_2, cnt) AS tv_show_genre_2,
             arg_max(tv_show_genre_3, cnt) AS tv_show_genre_3
      FROM (
        SELECT tv_show_title, tv_show_category, tv_show_genre_1, tv_show_genre_2, tv_show_genre_3,
               count(*) OVER (PARTITION BY tv_show_title, tv_show_category, tv_show_genre_1,
                                              tv_show_genre_2, tv_show_genre_3) AS cnt
        FROM {union}
        WHERE tv_show_title IS NOT NULL AND trim(tv_show_title) <> ''
      ) GROUP BY tv_show_title
    """)


def enrich(con, tables):
    """Дополняем категории/жанры из справочника: сначала по id, потом по названию."""
    for tbl in tables:
        for src, key in [("meta_by_id", "tv_show_id"), ("meta_by_title", "tv_show_title")]:
            con.execute(f"""
              UPDATE {tbl} SET
                tv_show_category = COALESCE(NULLIF({tbl}.tv_show_category,''), m.tv_show_category),
                tv_show_genre_1  = COALESCE(NULLIF({tbl}.tv_show_genre_1,''),  m.tv_show_genre_1),
                tv_show_genre_2  = COALESCE(NULLIF({tbl}.tv_show_genre_2,''),  m.tv_show_genre_2),
                tv_show_genre_3  = COALESCE(NULLIF({tbl}.tv_show_genre_3,''),  m.tv_show_genre_3)
              FROM {src} m WHERE m.{key} = {tbl}.{key}
            """)


# ============================================================ метрика
def map_at_k(preds, truth, k=5, require_full=False):
    """MAP@k.

    Для юзера u:
        R_u    = множество релевантных программ (его Top-k по частоте в validation)
        hits_i = |R_u ∩ {pred_1..pred_i}|
        AP@k   = (1/min(k, |R_u|)) * Σ_{i=1..k} [pred_i ∈ R_u] * hits_i / i
                 (среднее Precision@i по РЕЛЕВАНТНЫМ позициям)
        MAP@k  = среднее AP@k по юзерам.

    require_full=True  — только юзеры с |R_u| == k (строгий ground truth из 5 программ);
    require_full=False — все юзеры, нормировка min(k, |R_u|) (мягкий вариант).
    """
    aps, n_users, n_skip = [], 0, 0
    for u, rel in truth.items():
        if require_full and len(rel) < k:
            n_skip += 1
            continue
        n_users += 1
        rel_set = set(rel)
        hits, s = 0, 0.0
        for i, item in enumerate(preds.get(u, [])[:k], start=1):
            if item in rel_set:
                hits += 1
                s += hits / i
        aps.append(s / min(k, len(rel_set)) if rel_set else 0.0)
    return (float(np.mean(aps)) if aps else 0.0), n_users, n_skip


def recall_at_k(preds, truth, k=5):
    hits = n = 0
    for u, rel in truth.items():
        n += 1
        hits += len(set(preds.get(u, [])[:k]) & set(rel))
    return hits / (n * k) if n else 0.0


def coverage(preds):
    return len({i for v in preds.values() for i in v})


def selftest():
    """Юнит-тесты MAP@5 на эталонных значениях."""
    cases = [
        ("идеальное попадание", {1: [1, 2, 3, 4, 5]}, {1: [1, 2, 3, 4, 5]}, 1.0, False),
        ("нет совпадений", {1: [6, 7, 8, 9, 10]}, {1: [1, 2, 3, 4, 5]}, 0.0, False),
        ("релевантная на позиции 1", {1: [1, 7, 8, 9, 10]}, {1: [1, 2, 3, 4, 5]}, 0.2, False),
        ("релевантная на позиции 3", {1: [7, 8, 1, 9, 10]}, {1: [1, 2, 3, 4, 5]}, 1 / 15, False),
        ("релевантные на позициях 1 и 3", {1: [1, 7, 2, 9, 10]}, {1: [1, 2, 3, 4, 5]}, (1 + 2 / 3) / 5, False),
        ("релевантные на позициях 1 и 2", {1: [1, 2, 8, 9, 10]}, {1: [1, 2, 3, 4, 5]}, 2 / 5, False),
        ("мягкий: |R|=2, обе найдены", {1: [1, 2]}, {1: [1, 2]}, 1.0, False),
        ("мягкий: |R|=2, одна на позиции 1", {1: [1, 8]}, {1: [1, 2]}, 0.5, False),
        ("среднее по двум юзерам", {1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9, 10]},
         {1: [1, 2, 3, 4, 5], 2: [1, 2, 3, 4, 5]}, 0.5, False),
    ]
    ok = True
    print("--- юнит-тесты MAP@5 ---")
    for name, preds, truth, expected, strict in cases:
        m, _, _ = map_at_k(preds, truth, 5, strict)
        good = abs(m - expected) < 1e-9
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {name:<32} {m:.6f} (ожидалось {expected:.6f})")
    m, n, skipped = map_at_k({1: [1, 2]}, {1: [1, 2]}, 5, True)
    good = (n == 0 and skipped == 1)
    ok &= good
    print(f"  [{'OK ' if good else 'FAIL'}] строгий режим: юзер с |R|=2 отброшен")
    print("ИТОГ:", "все тесты пройдены" if ok else "ЕСТЬ ОШИБКИ")
    return ok


# ============================================================ утилиты
def topk_from_scores(scores, users, item_ids, k=5, mask=None, exclude=None):
    """Top-k по матрице оценок (n_users x n_items) -> {user_id: [item_id, ...]}.

    mask    — 1-D bool по item: True = item разрешён.
    exclude — 2-D bool (n_users, n_items): True = item запрещён конкретному юзеру.
    """
    out = {}
    ids_arr = np.asarray(item_ids, dtype=np.int64)
    kk = min(k, scores.shape[1])
    for r, u in enumerate(users):
        row = scores[r].astype(np.float64, copy=True)
        if mask is not None:
            row[~mask] = -np.inf
        if exclude is not None:
            row[exclude[r]] = -np.inf
        idx = np.argpartition(-row, kk - 1)[:kk]
        idx = idx[np.lexsort((ids_arr[idx], -row[idx]))]   # детерминированный tie-break
        out[u] = [int(ids_arr[i]) for i in idx[:k] if np.isfinite(row[i])]
    return out


def make_truth(con, agg_table, k=5):
    """Ground truth: Top-k программ юзера по частоте (n_airings) в validation.

    ВАЖНО: внешний ORDER BY нужен, иначе строки возвращаются в произвольном порядке
    (окно window function не задаёт порядок вывода), а список должен идти
    по убыванию частоты.
    """
    rows = con.execute(f"""
        SELECT user_id, tv_show_id FROM (
            SELECT user_id, tv_show_id, n_airings, max_screen_time,
                   row_number() OVER (PARTITION BY user_id
                                      ORDER BY n_airings DESC, max_screen_time DESC,
                                               tv_show_id ASC) AS rn
            FROM {agg_table}
        ) WHERE rn <= {k}
        ORDER BY user_id, rn
    """).fetchall()
    truth = defaultdict(list)
    for u, it in rows:
        truth[int(u)].append(int(it))
    return dict(truth)


# ============================================================ модели
def model_popularity(con, users, agg_table, k=5):
    """Глобальный Top-k по частоте в train (одинаковый список для всех юзеров)."""
    rows = con.execute(f"""
        SELECT tv_show_id, sum(n_airings) AS freq
        FROM {agg_table} GROUP BY tv_show_id
        ORDER BY freq DESC, tv_show_id ASC
    """).fetchall()
    top = [int(r[0]) for r in rows[:k]]
    return {u: list(top) for u in users}, top


def model_als(con, users, item_ids, agg_table, args, value_mode="airings",
              all_users=None, filter_liked=False, candidate_mask=None, verbose=True):
    """ALS через implicit на матрице user x item."""
    all_users = all_users or users
    rows = con.execute(f"""
        SELECT user_id, tv_show_id,
               CASE WHEN '{value_mode}' = 'airings' THEN n_airings ELSE max_screen_time END AS v
        FROM {agg_table} WHERE v > 0
    """).fetchall()

    user_index = {u: i for i, u in enumerate(all_users)}
    item_index = {it: i for i, it in enumerate(item_ids)}
    r = np.array([user_index[x[0]] for x in rows], dtype=np.int32)
    c = np.array([item_index[x[1]] for x in rows], dtype=np.int32)
    v = np.array([float(x[2]) for x in rows], dtype=np.float32)
    matrix = sp.csr_matrix((v, (r, c)), shape=(len(all_users), len(item_ids)))
    if verbose:
        print(f"    матрица {matrix.shape}, nnz={matrix.nnz}, "
              f"плотность={matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.3%}")

    model = AlternatingLeastSquares(factors=args.factors, iterations=args.iterations,
                                    regularization=args.regularization,
                                    random_state=args.seed, use_gpu=False)
    model.fit(matrix, show_progress=False)

    uf = np.asarray(model.user_factors)[[user_index[u] for u in users]]
    scores = uf @ np.asarray(model.item_factors).T

    exclude = None
    if filter_liked:
        exclude = np.zeros((len(users), len(item_ids)), dtype=bool)
        eval_pos = {u: i for i, u in enumerate(users)}
        for x in rows:
            p = eval_pos.get(x[0])
            if p is not None:
                exclude[p, item_index[x[1]]] = True

    return topk_from_scores(scores, users, item_ids, k=5,
                            mask=candidate_mask, exclude=exclude)


def model_cbf(con, users, agg_table, candidates_table, all_users=None, k=5,
              pop_weight=0.1, value_mode="airings", filter_liked=False, verbose=True):
    """Content-based: профиль юзера = распределение весов по категориям и жанрам."""
    all_users = all_users or users
    prof_rows = con.execute(f"""
        SELECT p.user_id, f.dim, sum(p.v) AS w
        FROM (
            SELECT user_id, tv_show_id, tv_show_category,
                   tv_show_genre_1, tv_show_genre_2, tv_show_genre_3,
                   CASE WHEN '{value_mode}' = 'airings' THEN n_airings ELSE max_screen_time END AS v
            FROM {agg_table}
        ) p
        JOIN LATERAL (
            SELECT 'cat:' || tv_show_category AS dim
              WHERE tv_show_category IS NOT NULL AND trim(tv_show_category) <> ''
            UNION ALL SELECT 'g1:' || tv_show_genre_1
              WHERE tv_show_genre_1 IS NOT NULL AND trim(tv_show_genre_1) <> ''
            UNION ALL SELECT 'g2:' || tv_show_genre_2
              WHERE tv_show_genre_2 IS NOT NULL AND trim(tv_show_genre_2) <> ''
            UNION ALL SELECT 'g3:' || tv_show_genre_3
              WHERE tv_show_genre_3 IS NOT NULL AND trim(tv_show_genre_3) <> ''
        ) f ON TRUE
        GROUP BY 1, 2
    """).fetchall()

    dims = sorted({r[1] for r in prof_rows})
    dim_index = {d: i for i, d in enumerate(dims)}
    user_index = {u: i for i, u in enumerate(all_users)}
    pr = np.array([user_index[r[0]] for r in prof_rows], dtype=np.int32)
    pc = np.array([dim_index[r[1]] for r in prof_rows], dtype=np.int32)
    pv = np.array([float(r[2]) for r in prof_rows], dtype=np.float32)

    U = np.zeros((len(all_users), len(dims)), dtype=np.float32)
    for (r, c), val in zip(zip(pr, pc), pv):
        U[r, c] = val
    norms = np.sqrt((U ** 2).sum(axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    U /= norms                                        # L2-нормализация профиля

    cand_rows = con.execute(f"""
        SELECT tv_show_id, tv_show_category, tv_show_genre_1, tv_show_genre_2, tv_show_genre_3
        FROM {candidates_table}""").fetchall()
    cand_ids = [int(r[0]) for r in cand_rows]
    V = np.zeros((len(cand_ids), len(dims)), dtype=np.float32)
    for i, r in enumerate(cand_rows):
        for pref, val in [("cat:", r[1]), ("g1:", r[2]), ("g2:", r[3]), ("g3:", r[4])]:
            if val is None or str(val).strip() == "":
                continue
            d = pref + str(val).strip()
            if d in dim_index:
                V[i, dim_index[d]] = 1.0
    vnorms = np.sqrt((V ** 2).sum(axis=1, keepdims=True))
    vnorms[vnorms == 0] = 1.0
    V /= vnorms                                        # L2-нормализация признаков программы

    scores = U @ V.T                                   # косинус(профиль, программа)
    # prior по популярности: разрывает ничьи (у многих программ одинаковый набор жанров)
    pop_map = {int(r[0]): float(r[1]) for r in con.execute(f"""
        SELECT tv_show_id, sum(n_airings) FROM {agg_table} GROUP BY tv_show_id""").fetchall()}
    pop = np.array([pop_map.get(c, 0.0) for c in cand_ids], dtype=np.float32)
    pmax = pop.max() if pop.max() > 0 else 1.0
    scores = scores + pop_weight * (pop / pmax)[None, :]

    cand_arr = np.array(cand_ids, dtype=np.int64)
    cand_index = {int(c): i for i, c in enumerate(cand_arr)}
    eval_pos = {u: i for i, u in enumerate(users)}
    exclude = None
    if filter_liked:
        exclude = np.zeros((len(users), len(cand_arr)), dtype=bool)
        for u, it in con.execute(f"SELECT user_id, tv_show_id FROM {agg_table}").fetchall():
            p, c = eval_pos.get(int(u)), cand_index.get(int(it))
            if p is not None and c is not None:
                exclude[p, c] = True

    out = {}
    kk = min(k, len(cand_arr))
    for u in users:
        row = scores[user_index[u]].astype(np.float64, copy=True)
        if exclude is not None:
            row[exclude[eval_pos[u]]] = -np.inf
        idx = np.argpartition(-row, kk - 1)[:kk]
        idx = idx[np.lexsort((cand_arr[idx], -row[idx]))]
        out[u] = [int(cand_arr[i]) for i in idx[:k] if np.isfinite(row[i])]
    if verbose:
        print(f"    CBF: {len(dims)} признаков (категории+жанры), {len(cand_ids)} кандидатов")
    return out


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--factors", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--regularization", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cbf-pop-weight", type=float, default=1.0)
    ap.add_argument("--als-value", default="airings", choices=["airings", "screen_time"])
    ap.add_argument("--filter-liked", action="store_true",
                    help="не рекомендовать программы, уже просмотренные юзером в train")
    ap.add_argument("--no-sweep", action="store_true", help="не подбирать гиперпараметры")
    ap.add_argument("--selftest", action="store_true", help="прогнать только юнит-тесты MAP@5")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys_exit(0 if selftest() else 1)

    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='8GB';")
    t_all = time.time()

    # ---------------- данные и внешний сплит
    split = load_data(con, args.data_dir)
    build_metadata(con)
    build_interactions(con, "train", "sched_train", f"l.session_start < TIMESTAMP '{split}'", args.threshold)
    build_interactions(con, "valid", "sched_valid", f"l.session_start >= TIMESTAMP '{split}'", args.threshold)

    con.execute("""
      CREATE OR REPLACE TABLE cand_valid AS
      SELECT tv_show_id, any_value(tv_show_title) AS tv_show_title,
             any_value(tv_show_category) AS tv_show_category,
             any_value(tv_show_genre_1) AS tv_show_genre_1,
             any_value(tv_show_genre_2) AS tv_show_genre_2,
             any_value(tv_show_genre_3) AS tv_show_genre_3
      FROM sched_valid WHERE tv_show_id > 0 GROUP BY tv_show_id
    """)
    enrich(con, ["train_agg", "cand_valid"])

    print(f"граница периодов SPLIT = {split}")
    for name in ["train", "valid"]:
        raw = con.execute(f"SELECT count(*) FROM {name}_int").fetchone()[0]
        agg = con.execute(f"SELECT count(*) FROM {name}_agg").fetchone()[0]
        u = con.execute(f"SELECT count(DISTINCT user_id) FROM {name}_agg").fetchone()[0]
        s = con.execute(f"SELECT count(DISTINCT tv_show_id) FROM {name}_agg").fetchone()[0]
        print(f"[{name}] эфиров {raw:,} | пар (user,show) {agg:,} | юзеров {u:,} | программ {s:,}")
    print(f"[кандидаты] программ в эфире 21–30: "
          f"{con.execute('SELECT count(*) FROM cand_valid').fetchone()[0]:,}")

    truth = make_truth(con, "valid_agg", 5)
    n_lt5 = sum(1 for v in truth.values() if len(v) < 5)
    print(f"[ground truth] юзеров {len(truth)}, из них с <5 программами {n_lt5}")

    users = sorted({int(r[0]) for r in con.execute("SELECT DISTINCT user_id FROM valid_agg").fetchall()})
    train_users = {int(r[0]) for r in con.execute("SELECT DISTINCT user_id FROM train_agg").fetchall()}
    all_users = sorted(set(users) | train_users)
    items = sorted({int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM train_agg").fetchall()})
    cand_ids = {int(r[0]) for r in con.execute("SELECT tv_show_id FROM cand_valid").fetchall()}
    cold = [u for u in users if u not in train_users]
    cand_mask = np.array([it in cand_ids for it in items], dtype=bool)
    print(f"[пространство] юзеров {len(users)} (cold без train: {len(cold)}), "
          f"программ train {len(items)}, кандидатов {len(cand_ids)}, "
          f"train ∩ кандидаты {int(cand_mask.sum())}")

    # ---------------- подбор гиперпараметров на ВНУТРЕННЕМ сплите
    if not args.no_sweep:
        t0 = time.time()
        args = sweep_hyperparams(con, args, split)
        print(f"[подбор гиперпараметров на внутреннем сплите] {time.time()-t0:.1f}s")

    # ---------------- финальные модели на внешнем сплите
    print("\n=== Popularity ===")
    pop_preds, pop_top = model_popularity(con, users, "train_agg")
    print(f"    глобальный Top-5: {pop_top}")

    print("\n=== ALS (implicit) ===")
    als_preds = model_als(con, users, items, "train_agg", args,
                          value_mode=args.als_value, all_users=all_users,
                          filter_liked=args.filter_liked, candidate_mask=cand_mask)
    other = "screen_time" if args.als_value == "airings" else "airings"
    als_preds_alt = model_als(con, users, items, "train_agg", args, value_mode=other,
                              all_users=all_users, filter_liked=args.filter_liked,
                              candidate_mask=cand_mask, verbose=False)

    print("\n=== Content-Based ===")
    cbf_preds = model_cbf(con, users, "train_agg", "cand_valid", all_users=all_users,
                          pop_weight=args.cbf_pop_weight, value_mode=args.als_value,
                          filter_liked=args.filter_liked)

    # cold-start юзеры (нет train-истории): ALS не может их обслужить -> fallback на популярность
    for preds in (als_preds, als_preds_alt, cbf_preds):
        for u in cold:
            if not preds.get(u):
                preds[u] = list(pop_top)

    # ---------------- метрики
    print("\n" + "=" * 84)
    print("СРАВНЕНИЕ MAP@5   (validation = недели 21–30, ground truth = Топ-5 по частоте)")
    print("=" * 84)
    rows = []
    for name, preds in [("Popularity (глобальный Top-5)", pop_preds),
                        (f"ALS (implicit, {args.als_value})", als_preds),
                        (f"ALS (implicit, {other})", als_preds_alt),
                        ("CBF (категории + жанры)", cbf_preds)]:
        m_strict, n_used, n_skip = map_at_k(preds, truth, 5, True)
        m_soft, _, _ = map_at_k(preds, truth, 5, False)
        rec = recall_at_k(preds, truth, 5)
        cov = coverage(preds)
        rows.append((name, m_strict, m_soft, rec, cov, n_used, n_skip))
        print(f"\n{name}")
        print(f"  MAP@5 строгий (|GT|=5): {m_strict:.4f}   юзеров {n_used}, отброшено {n_skip}")
        print(f"  MAP@5 мягкий (все)    : {m_soft:.4f}")
        print(f"  Recall@5              : {rec:.4f}")
        print(f"  Покрытие (уник. программ): {cov}")

    print("\n" + "=" * 84)
    print(f"{'Модель':<32}{'MAP@5 строгий':>16}{'MAP@5 мягкий':>16}{'Recall@5':>12}{'Покрытие':>11}")
    print("-" * 84)
    for name, m_strict, m_soft, rec, cov, n_used, n_skip in rows:
        print(f"{name:<32}{m_strict:>16.4f}{m_soft:>16.4f}{rec:>12.4f}{cov:>11}")
    print("-" * 84)
    print(f"юзеров в строгой оценке: {rows[0][5]}; отброшено (в validation <5 программ): {rows[0][6]}")

    # reachability — почему у ALS есть потолок
    gt_items = {i for v in truth.values() for i in v}
    reach = len(gt_items & cand_ids & set(items))
    print(f"\nдостижимость ground truth: программ в GT {len(gt_items)}, "
          f"из них в эфире 21–30 {len(gt_items & cand_ids)} "
          f"({len(gt_items & cand_ids)/len(gt_items):.1%}), "
          f"из них ещё и в train {reach} ({reach/len(gt_items):.1%}) — это потолок для ALS")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for tbl, name in [("train_agg", "train_agg"), ("valid_agg", "valid_agg"),
                          ("cand_valid", "candidates_valid")]:
            con.execute(f"COPY (SELECT * FROM {tbl}) TO '{args.out_dir}/{name}.parquet' "
                        f"(FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"\n[out] train_agg.parquet, valid_agg.parquet, candidates_valid.parquet -> {args.out_dir}")

    print(f"\nвсего {time.time()-t_all:.1f}s")
    con.close()


def sweep_hyperparams(con, args, split):
    """Подбор гиперпараметров ALS и веса CBF-приора на внутреннем сплите.

    Внутренний сплит: недели 11–15 (train) / 16–20 (validation) — обе половины
    файла export_arh_11-20-final.csv. Отчётный validation (21–30) в подборе не участвует.
    """
    train_lo = con.execute("SELECT min(program_start) FROM sched_train").fetchone()[0]
    inner_split = train_lo + (split - train_lo) / 2
    print(f"внутренний сплит: train < {inner_split}, validation [{inner_split}, {split})")
    build_interactions(con, "inner_train", "sched_train",
                       f"l.session_start < TIMESTAMP '{inner_split}'", args.threshold)
    build_interactions(con, "inner_valid", "sched_train",
                       f"l.session_start >= TIMESTAMP '{inner_split}'", args.threshold)
    con.execute(f"""
      CREATE OR REPLACE TABLE cand_inner AS
      SELECT tv_show_id, any_value(tv_show_title) AS tv_show_title,
             any_value(tv_show_category) AS tv_show_category,
             any_value(tv_show_genre_1) AS tv_show_genre_1,
             any_value(tv_show_genre_2) AS tv_show_genre_2,
             any_value(tv_show_genre_3) AS tv_show_genre_3
      FROM sched_train WHERE tv_show_id > 0 AND program_start >= TIMESTAMP '{inner_split}'
      GROUP BY tv_show_id
    """)
    enrich(con, ["inner_train_agg", "cand_inner"])

    truth = make_truth(con, "inner_valid_agg", 5)
    users = sorted({int(r[0]) for r in con.execute("SELECT DISTINCT user_id FROM inner_valid_agg").fetchall()})
    train_users = {int(r[0]) for r in con.execute("SELECT DISTINCT user_id FROM inner_train_agg").fetchall()}
    all_users = sorted(set(users) | train_users)
    items = sorted({int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM inner_train_agg").fetchall()})
    cand_ids = {int(r[0]) for r in con.execute("SELECT tv_show_id FROM cand_inner").fetchall()}
    cand_mask = np.array([it in cand_ids for it in items], dtype=bool)

    best = (None, None)
    grid = [(f, i, r) for f in (4, 8, 16) for i in (10, 20) for r in (0.1, 1.0, 10.0)]
    print("    ALS:")
    for f, i, r in grid:
        a = argparse.Namespace(factors=f, iterations=i, regularization=r, seed=args.seed)
        p = model_als(con, users, items, "inner_train_agg", a, value_mode="airings",
                      all_users=all_users, candidate_mask=cand_mask, verbose=False)
        m, _, _ = map_at_k(p, truth, 5, True)
        print(f"      factors={f:<3} iters={i:<3} reg={r:<5} -> MAP@5={m:.4f}")
        if best[0] is None or m > best[0]:
            best = (m, (f, i, r))
    args.factors, args.iterations, args.regularization = best[1]
    print(f"    выбрано: factors={args.factors} iters={args.iterations} reg={args.regularization} "
          f"(внутренний MAP@5={best[0]:.4f})")

    best_w = (None, None)
    print("    CBF (вес popularity-приора):")
    for w in (0.0, 0.1, 0.3, 1.0, 3.0):
        p = model_cbf(con, users, "inner_train_agg", "cand_inner", all_users=all_users,
                      pop_weight=w, verbose=False)
        m, _, _ = map_at_k(p, truth, 5, True)
        print(f"      pop_weight={w:<5} -> MAP@5={m:.4f}")
        if best_w[0] is None or m > best_w[0]:
            best_w = (m, w)
    args.cbf_pop_weight = best_w[1]
    print(f"    выбрано: cbf_pop_weight={args.cbf_pop_weight} "
          f"(внутренний MAP@5={best_w[0]:.4f})")
    return args


def sys_exit(code):
    raise SystemExit(code)


if __name__ == "__main__":
    main()
