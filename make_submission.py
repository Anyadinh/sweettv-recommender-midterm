#!/usr/bin/env python3
"""
Финальный submission.csv для тестовых пользователей.

Подход: ALS (implicit) на n_airings — лучший метод по MAP@5 на валидации недель 21–30.

  * Обучение   — ВСЕ тренировочные данные: недели 11–30 (расписание 11-20 + 21-30),
                 screen_time >= 0.8, tv_show_id > 0.
  * Кандидаты  — программы из export_arh_31-42-final.csv (тестовые недели 31–42).
  * Cold start — юзеры без единого взаимодействия: глобальный популярный Топ-5.
  * Формат    — user_id,tv_show_id где tv_show_id = 5 id через пробел (шаблон submission.csv).

Запуск:
  python3 make_submission.py
  python3 make_submission.py --template submission.csv --out submission.csv
"""

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recsys_models as R


def read_template(path):
    """Читаем шаблон: список user_id тестовых юзеров (в исходном порядке)."""
    users = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            if row and row[0].strip():
                users.append(row[0].strip())
    print(f"шаблон {path}: {len(users)} тестовых юзеров, заголовок: {header}")
    return users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--template", default=None, help="шаблон (по умолчанию <data-dir>/submission.csv)")
    ap.add_argument("--out", default=None, help="куда сохранить (по умолчанию <data-dir>/submission.csv)")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--factors", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--regularization", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--als-value", default="airings", choices=["airings", "screen_time"])
    ap.add_argument("--report-user", type=int, default=None, help="user_id для отчёта (с историей)")
    ap.add_argument("--report-cold-user", type=int, default=None)
    args = ap.parse_args()

    data_dir = args.data_dir
    template = args.template or os.path.join(data_dir, "submission.csv")
    out_path = args.out or os.path.join(data_dir, "submission.csv")
    t_all = time.time()

    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='8GB';")

    # ---------------------------------------------------------- данные
    split = R.load_data(con, data_dir)
    R.build_metadata(con)

    # тренировка = недели 11–30: две половины, чтобы не было двойного счёта на границе
    R.build_interactions(con, "train", "sched_train",
                         f"l.session_start < TIMESTAMP '{split}'", args.threshold)
    R.build_interactions(con, "valid", "sched_valid",
                         f"l.session_start >= TIMESTAMP '{split}'", args.threshold)
    con.execute("""
      CREATE OR REPLACE TABLE full_agg AS
      SELECT user_id, tv_show_id,
             sum(n_airings) AS n_airings,
             max(max_screen_time) AS max_screen_time,
             sum(total_overlap) AS total_overlap,
             any_value(tv_show_title) AS tv_show_title,
             any_value(tv_show_category) AS tv_show_category,
             any_value(tv_show_genre_1) AS tv_show_genre_1,
             any_value(tv_show_genre_2) AS tv_show_genre_2,
             any_value(tv_show_genre_3) AS tv_show_genre_3
      FROM (SELECT * FROM train_agg UNION ALL SELECT * FROM valid_agg)
      GROUP BY user_id, tv_show_id
    """)
    R.enrich(con, ["full_agg"])

    print(f"граница периодов SPLIT = {split}")
    for name in ["train", "valid"]:
        raw = con.execute(f"SELECT count(*) FROM {name}_int").fetchone()[0]
        agg = con.execute(f"SELECT count(*) FROM {name}_agg").fetchone()[0]
        print(f"  [{name}] эфиров {raw:,} | пар (user,show) {agg:,}")
    n_full = con.execute("SELECT count(*) FROM full_agg").fetchone()[0]
    n_full_users = con.execute("SELECT count(DISTINCT user_id) FROM full_agg").fetchone()[0]
    n_full_items = con.execute("SELECT count(DISTINCT tv_show_id) FROM full_agg").fetchone()[0]
    print(f"  [train 11–30] пар (user,show) {n_full:,} | юзеров {n_full_users:,} | программ {n_full_items:,}")

    # ---------------------------------------------------------- кандидаты: недели 31–42
    con.execute("""
      CREATE OR REPLACE TABLE cand_test AS
      SELECT tv_show_id,
             any_value(tv_show_title) AS tv_show_title,
             any_value(tv_show_category) AS tv_show_category,
             any_value(tv_show_genre_1) AS tv_show_genre_1,
             any_value(tv_show_genre_2) AS tv_show_genre_2,
             any_value(tv_show_genre_3) AS tv_show_genre_3,
             min(program_start) AS first_airing
      FROM sched_test WHERE tv_show_id > 0 GROUP BY tv_show_id
    """)
    R.enrich(con, ["cand_test"])
    cand_ids = {int(r[0]) for r in con.execute("SELECT tv_show_id FROM cand_test").fetchall()}
    train_items = sorted({int(r[0]) for r in con.execute("SELECT DISTINCT tv_show_id FROM full_agg").fetchall()})
    reachable = sorted(cand_ids & set(train_items))
    print(f"  [кандидаты 31–42] программ в эфире {len(cand_ids):,}; "
          f"из них известных ALS (есть в train) {len(reachable):,} "
          f"({len(reachable)/len(cand_ids):.1%})")

    # ---------------------------------------------------------- ALS на всех train-данных
    all_users = sorted({int(r[0]) for r in con.execute("SELECT DISTINCT user_id FROM full_agg").fetchall()})
    train_user_set = set(all_users)
    items = train_items
    item_index = {it: i for i, it in enumerate(items)}
    cand_mask = np.array([it in cand_ids for it in items], dtype=bool)

    rows = con.execute(f"""
        SELECT user_id, tv_show_id,
               CASE WHEN '{args.als_value}' = 'airings' THEN n_airings ELSE max_screen_time END AS v
        FROM full_agg WHERE v > 0
    """).fetchall()
    user_index = {u: i for i, u in enumerate(all_users)}
    import scipy.sparse as sp
    from implicit.als import AlternatingLeastSquares
    r = np.array([user_index[x[0]] for x in rows], dtype=np.int32)
    c = np.array([item_index[x[1]] for x in rows], dtype=np.int32)
    v = np.array([float(x[2]) for x in rows], dtype=np.float32)
    matrix = sp.csr_matrix((v, (r, c)), shape=(len(all_users), len(items)))
    print(f"\nматрица ALS: {matrix.shape}, nnz={matrix.nnz}, "
          f"плотность={matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.3%}")

    t0 = time.time()
    model = AlternatingLeastSquares(factors=args.factors, iterations=args.iterations,
                                    regularization=args.regularization,
                                    random_state=args.seed, use_gpu=False)
    model.fit(matrix, show_progress=False)
    print(f"ALS обучена: factors={args.factors}, iterations={args.iterations}, "
          f"reg={args.regularization}, value={args.als_value} ({time.time()-t0:.1f}s)")

    # ---------------------------------------------------------- популярный fallback
    pop_rows = con.execute("""
        SELECT tv_show_id, sum(n_airings) AS freq
        FROM full_agg GROUP BY tv_show_id ORDER BY freq DESC, tv_show_id ASC
    """).fetchall()
    pop_all = [int(x[0]) for x in pop_rows]
    pop_cand = [i for i in pop_all if i in cand_ids][:5]
    if len(pop_cand) < 5:                       # добираем из кандидатов по частоте в train
        seen = set(pop_cand)
        for i in pop_all:
            if i not in seen and i in cand_ids:
                pop_cand.append(i)
                if len(pop_cand) == 5:
                    break
    print(f"популярный Топ-5 (в эфире 31–42): {pop_cand}")

    # ---------------------------------------------------------- рекомендации
    test_users = read_template(template)
    test_int = [u for u in test_users if int(u) in train_user_set]
    cold_users = [u for u in test_users if int(u) not in train_user_set]
    print(f"тестовых юзеров: {len(test_users)}; с историей {len(test_int)}, "
          f"cold start {len(cold_users)}")

    # ALS-скоринг для тех, у кого есть история
    uf = np.asarray(model.user_factors)[[user_index[int(u)] for u in test_int]]
    scores = uf @ np.asarray(model.item_factors).T
    als_top = R.topk_from_scores(scores, [int(u) for u in test_int], items, k=5,
                                 mask=cand_mask)

    # добивка до 5 позиций популярными (если ALS не набрал 5 доступных кандидатов)
    def fill(top):
        out = list(top)
        for i in pop_cand:
            if len(out) == 5:
                break
            if i not in out:
                out.append(i)
        return out[:5]

    preds = {}
    for u in test_users:
        uid = int(u)
        if uid in als_top:
            preds[u] = fill(als_top[uid])
        else:
            preds[u] = list(pop_cand)

    # ---------------------------------------------------------- запись submission.csv
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "tv_show_id"])
        for u in test_users:
            w.writerow([u, " ".join(str(i) for i in preds[u])])
    os.replace(tmp_path, out_path)
    print(f"\nсохранено {len(test_users)} строк -> {out_path}")

    # ---------------------------------------------------------- проверки
    bad = [u for u in test_users if len(preds[u]) != 5 or any(i <= 0 for i in preds[u])]
    unknown = [u for u in test_users if any(i not in cand_ids for i in preds[u])]
    dupes = [u for u in test_users if len(set(preds[u])) != 5]
    print(f"проверки: строк с неверным числом/нулевыми id: {len(bad)}; "
          f"id вне эфира 31–42: {len(unknown)}; строк с дублями: {len(dupes)}")

    # ---------------------------------------------------------- отчёт по двум юзерам
    def show(user_id, title):
        uid = int(user_id)
        hist = con.execute(f"""
            SELECT count(*) n, sum(n_airings) a FROM full_agg WHERE user_id = {uid}""").fetchone()
        print(f"\n{'='*84}\n{title}: user_id = {user_id}\n{'='*84}")
        if hist[0]:
            top_hist = con.execute(f"""
                SELECT tv_show_id, n_airings, max_screen_time, tv_show_title, tv_show_category
                FROM full_agg WHERE user_id = {uid}
                ORDER BY n_airings DESC, max_screen_time DESC, tv_show_id LIMIT 5""").fetchall()
            print(f"история в train (11–30): {hist[0]} программ, {hist[1]} эфиров; "
                  f"топ по частоте: "
                  f"{[(x[0], x[1]) for x in top_hist]}")
        else:
            print("история в train (11–30): ПУСТО -> cold start, выдан популярный Топ-5")
        print("Рекомендации Топ-5:")
        for pos, i in enumerate(preds[str(user_id)], 1):
            row = con.execute(f"""
                SELECT tv_show_title, tv_show_category, first_airing
                FROM cand_test WHERE tv_show_id = {i}""").fetchone()
            title_s, cat, air = row if row else (None, None, None)
            extra = ""
            if hist[0]:
                h = con.execute(f"""
                    SELECT n_airings, max_screen_time FROM full_agg
                    WHERE user_id = {uid} AND tv_show_id = {i}""").fetchone()
                extra = (f"   [в train: {h[0]} эфиров, screen_time={h[1]:.2f}]"
                         if h else "   [в train не смотрел]")
            print(f"  {pos}. tv_show_id={i:<10} {str(title_s)[:58]:<58} "
                  f"категория: {str(cat)[:22]:<22}{extra}")

    # автовыбор: юзер с самой длинной историей и первый cold-start
    if not args.report_user:
        args.report_user = int(con.execute(f"""
            SELECT user_id FROM full_agg
            WHERE user_id IN ({','.join(str(int(u)) for u in test_int)})
            GROUP BY user_id ORDER BY sum(n_airings) DESC, user_id ASC LIMIT 1
        """).fetchone()[0])
    if not args.report_cold_user and cold_users:
        args.report_cold_user = int(cold_users[0])

    show(args.report_user, "Пользователь с историей")
    if args.report_cold_user:
        show(args.report_cold_user, "Пользователь без истории (cold start)")

    print(f"\nвсего {time.time()-t_all:.1f}s")
    con.close()


if __name__ == "__main__":
    main()
