#!/usr/bin/env python3
"""
Сборка interactions.parquet для квиза по RecSys (SweetTV TV Program Recommender).

Логи:        dataset11-30.csv        (user_id, vsetv_id, start_time, stop_time, duraton)
Расписание: export_arh_*-final.csv  (channel_id, start_time, duration, tv_show_id, ...)

Логика:
  1) сессия просмотра [log_start, log_stop) пересекается с программой
     [prog_start, prog_start + duration) на том же канале;
  2) overlap_seconds = max(0, min(log_stop, prog_end) - max(log_start, prog_start));
  3) screen_time = overlap_seconds / duration  (доля программы, которую досидел юзер);
  4) оставляем screen_time >= 0.8.

Запуск:
  python3 build_interactions.py                      # дефолтные пути
  python3 build_interactions.py --data-dir ~/Desktop/sweettv-tv-program-recommender
  python3 build_interactions.py --threshold 0.8 --out interactions.parquet
"""

import argparse
import glob
import os
import sys
import time

try:
    import duckdb
except ImportError:
    sys.exit("Нужен duckdb: pip install duckdb")


def find_data_files(data_dir):
    """Ищем логи и расписание с учётом возможных вариантов имени."""
    log_candidates = [
        "dataset11-30week.csv",
        "dataset11-30.csv",
        "dataset11-30week*.csv",
        "dataset*30*.csv",
    ]
    log_path = None
    for pat in log_candidates:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            log_path = hits[0]
            break
    if log_path is None:
        sys.exit(f"Не найден файл логов в {data_dir} (искал {log_candidates})")

    sched_paths = sorted(glob.glob(os.path.join(data_dir, "export_arh_*.csv")))
    if not sched_paths:
        sys.exit(f"Не найдены файлы расписания export_arh_*.csv в {data_dir}")
    return log_path, sched_paths


# --- схемы: читаем всё как VARCHAR и парсим даты сами (DuckDB не должен угадывать формат) ---
LOG_SCHEMA = (
    "{'user_id':'VARCHAR','vsetv_id':'VARCHAR','start_time':'VARCHAR',"
    "'stop_time':'VARCHAR','duraton':'VARCHAR'}"
)
SCHED_SCHEMA = (
    "{'channel_id':'VARCHAR','channel_title':'VARCHAR','start_time':'VARCHAR',"
    "'duration':'VARCHAR','tv_show_title':'VARCHAR','tv_show_id':'VARCHAR',"
    "'tv_show_category':'VARCHAR','tv_show_genre_1':'VARCHAR',"
    "'tv_show_genre_2':'VARCHAR','tv_show_genre_3':'VARCHAR',"
    "'year_of_production':'VARCHAR','director':'VARCHAR','actors':'VARCHAR'}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None, help="путь к parquet (по умолчанию <data-dir>/interactions.parquet)")
    ap.add_argument("--threshold", type=float, default=0.8, help="порог screen_time")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--memory-limit", default="8GB")
    ap.add_argument("--keep-zero-show-id", action="store_true",
                    help="не выкидывать строки с tv_show_id = 0 (в расписании это ~90%% строк)")
    ap.add_argument("--dedup", action="store_true",
                    help="свернуть до одной строки на (user_id, tv_show_id) — матрица для RecSys")
    args = ap.parse_args()

    log_path, sched_paths = find_data_files(args.data_dir)
    out_path = args.out or os.path.join(args.data_dir, "interactions.parquet")

    con = duckdb.connect()
    con.execute(f"SET threads={args.threads};")
    con.execute(f"SET memory_limit='{args.memory_limit}';")

    print(f"Логи:      {log_path}")
    for p in sched_paths:
        print(f"Расписание: {p}")
    print(f"Порог screen_time: {args.threshold}\n")

    # ------------------------------------------------------------------ 1. логи
    t0 = time.time()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE logs AS
        SELECT
            try_cast(user_id AS HUGEINT)              AS user_id,
            try_cast(vsetv_id AS BIGINT)              AS channel_id,
            strptime(start_time, '%Y-%m-%d %H:%M:%S') AS session_start,
            strptime(stop_time,  '%Y-%m-%d %H:%M:%S') AS session_stop,
            try_cast(duraton AS BIGINT)               AS log_duration
        FROM read_csv('{log_path}', header=true, quote='"',
                      columns={LOG_SCHEMA}, sample_size=-1)
        WHERE start_time IS NOT NULL AND stop_time IS NOT NULL
          AND strptime(stop_time, '%Y-%m-%d %H:%M:%S') > strptime(start_time, '%Y-%m-%d %H:%M:%S')
        """
    )
    n_logs = con.execute("SELECT count(*) FROM logs").fetchone()[0]
    print(f"[1] логи: {n_logs:,} строк ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ 2. расписание
    t0 = time.time()
    parts = []
    for p in sched_paths:
        parts.append(
            f"""
            SELECT
                try_cast(channel_id AS BIGINT)              AS channel_id,
                channel_title,
                strptime(start_time, '%d.%m.%Y %H:%M:%S')   AS program_start,
                try_cast(duration AS BIGINT)                AS duration,
                strptime(start_time, '%d.%m.%Y %H:%M:%S')
                    + try_cast(duration AS BIGINT) * INTERVAL 1 SECOND AS program_end,
                try_cast(tv_show_id AS BIGINT)              AS tv_show_id,
                tv_show_title,
                tv_show_category,
                tv_show_genre_1, tv_show_genre_2, tv_show_genre_3
            FROM read_csv('{p}', header=true, quote='"',
                          columns={SCHED_SCHEMA}, sample_size=-1)
            """
        )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE schedule AS
        SELECT * FROM ({" UNION ALL BY NAME ".join(parts)})
        WHERE duration IS NOT NULL AND duration > 0
          AND program_start IS NOT NULL
        """
    )
    n_sched = con.execute("SELECT count(*) FROM schedule").fetchone()[0]
    print(f"[2] расписание: {n_sched:,} строк ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ 3. пересечение
    t0 = time.time()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE ov AS
        SELECT
            l.user_id,
            l.channel_id,
            p.tv_show_id,
            p.tv_show_title,
            p.tv_show_category,
            p.tv_show_genre_1,
            p.tv_show_genre_2,
            p.tv_show_genre_3,
            l.session_start,
            l.session_stop,
            p.program_start,
            p.program_end,
            p.duration,
            GREATEST(0, date_diff('second',
                        GREATEST(l.session_start, p.program_start),
                        LEAST(l.session_stop,  p.program_end)))  AS overlap_seconds
        FROM logs l
        JOIN schedule p
          ON l.channel_id = p.channel_id
         AND l.session_start < p.program_end      -- сессия начинается до конца программы
         AND l.session_stop  > p.program_start     -- сессия заканчивается после начала программы
        """
    )
    n_ov = con.execute("SELECT count(*) FROM ov").fetchone()[0]
    print(f"[3] пар «сессия × программа» с пересечением по времени: {n_ov:,} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ 4. фильтр screen_time
    t0 = time.time()
    con.execute(
        f"""
        CREATE OR REPLACE TABLE interactions AS
        SELECT
            user_id, channel_id, tv_show_id, tv_show_title, tv_show_category,
            tv_show_genre_1, tv_show_genre_2, tv_show_genre_3,
            session_start, session_stop, program_start, program_end, duration,
            overlap_seconds,
            overlap_seconds::DOUBLE / duration::DOUBLE AS screen_time
        FROM ov
        WHERE overlap_seconds > 0
          AND overlap_seconds::DOUBLE / duration::DOUBLE >= {args.threshold}
        """
    )
    n_int = con.execute("SELECT count(*) FROM interactions").fetchone()[0]
    print(f"[4] interactions (screen_time >= {args.threshold}): {n_int:,} ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ 5. сохранение
    t0 = time.time()
    # user_id -> VARCHAR: ID не помещаются в INT64 (до ~1.8e19), а DuckDB пишет HUGEINT
    # в parquet как DOUBLE и теряет точность. Строка-идентификатор безопаснее.
    if args.dedup:
        # одна строка на (user_id, tv_show_id): берём самый "досмотренный" эфир
        sel = """
            SELECT CAST(user_id AS VARCHAR) AS user_id,
                   tv_show_id,
                   arg_max(tv_show_title, screen_time)   AS tv_show_title,
                   arg_max(tv_show_category, screen_time) AS tv_show_category,
                   arg_max(tv_show_genre_1, screen_time) AS tv_show_genre_1,
                   arg_max(tv_show_genre_2, screen_time) AS tv_show_genre_2,
                   arg_max(tv_show_genre_3, screen_time) AS tv_show_genre_3,
                   arg_max(channel_id, screen_time)       AS channel_id,
                   count(*)                               AS n_airings,
                   max(screen_time)                       AS screen_time,
                   max(overlap_seconds)                   AS overlap_seconds,
                   max(duration)                          AS duration,
                   min(session_start)                     AS first_session_start,
                   max(session_stop)                      AS last_session_stop
            FROM interactions
            {where}
            GROUP BY user_id, tv_show_id
        """
    else:
        sel = """
            SELECT CAST(user_id AS VARCHAR) AS user_id,
                   channel_id, tv_show_id, tv_show_title, tv_show_category,
                   tv_show_genre_1, tv_show_genre_2, tv_show_genre_3,
                   session_start, session_stop, program_start, program_end, duration,
                   overlap_seconds, screen_time
            FROM interactions
            {where}
        """
    where = "" if args.keep_zero_show_id else "WHERE tv_show_id > 0"
    con.execute(
        f"COPY ({sel.format(where=where)} ORDER BY user_id, tv_show_id) "
        f"TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    n_out = con.execute(
        f"SELECT count(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]
    print(f"[5] сохранено {n_out:,} строк -> {out_path} ({time.time()-t0:.1f}s)\n")

    # ------------------------------------------------------------------ 6. статистика
    def q(sql):
        return con.execute(sql).fetchall()

    print("=== Статистика ===")
    print("пользователей:", q("SELECT count(DISTINCT user_id) FROM interactions")[0][0])
    print("round-trip user_id (parquet):", q(
        f"SELECT count(DISTINCT user_id), count(DISTINCT user_id::HUGEINT), "
        f"max(length(user_id)) FROM read_parquet('{out_path}')")[0],
        "(кол-во уникальных | после каста в HUGEINT | макс. длина строки)")
    print("программ (tv_show_id):", q("SELECT count(DISTINCT tv_show_id) FROM interactions")[0][0])
    print("каналов:", q("SELECT count(DISTINCT channel_id) FROM interactions")[0][0])
    print("всего взаимодействий:", n_int)
    print("уникальных (user_id, tv_show_id):",
          q("SELECT count(*) FROM (SELECT DISTINCT user_id, tv_show_id FROM interactions)")[0][0])
    print("screen_time: min/avg/max =", q(
        "SELECT round(min(screen_time),3), round(avg(screen_time),3), round(max(screen_time),3) "
        "FROM interactions")[0])
    print("строк с tv_show_id = 0:",
          q("SELECT count(*) FROM interactions WHERE tv_show_id = 0")[0][0])
    print("\nраспределение screen_time:")
    for row in q("""
        SELECT bucket, count(*) FROM (
            SELECT CASE
                     WHEN screen_time = 1.0            THEN '1.000 (смотрел целиком)'
                     WHEN screen_time >= 0.95          THEN '0.95-1.00'
                     WHEN screen_time >= 0.90          THEN '0.90-0.95'
                     WHEN screen_time >= 0.85          THEN '0.85-0.90'
                     ELSE '0.80-0.85'
                   END AS bucket
            FROM interactions
        ) GROUP BY 1 ORDER BY 1
    """):
        print(f"  {row[0]:<26} {row[1]:,}")

    print("\n=== Ручная проверка пересечения (первые 8 строк) ===")
    for row in q("""
        SELECT user_id, channel_id, tv_show_id, session_start, session_stop,
               program_start, program_end, duration, overlap_seconds, screen_time
        FROM interactions ORDER BY user_id, program_start LIMIT 8
    """):
        (uid, ch, show, s0, s1, p0, p1, dur, ov, st) = row
        lo = max(s0, p0)
        hi = min(s1, p1)
        print(f"  user={uid} ch={ch} show={show}")
        print(f"    сессия  [{s0} .. {s1}]")
        print(f"    программа[{p0} .. {p1}]  duration={dur}s")
        print(f"    max(start)={lo}  min(stop)={hi}  ->  overlap = {hi} - {lo} = "
              f"{int((hi-lo).total_seconds())}s ; screen_time = {int((hi-lo).total_seconds())}/{dur} = {st:.3f}")

    con.close()


if __name__ == "__main__":
    main()
