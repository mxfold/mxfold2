#!/usr/bin/env python3
"""Build manuscript tables and a portable writing kit from local evidence."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import zipfile
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLES = HERE / 'tables'
SOURCE_EXT = {'.py', '.cpp', '.hpp', '.h', '.ipp', '.c', '.toml', '.yaml', '.yml', '.md', '.sh'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_tsv(path, rows, fields=None):
    if not fields:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    return list(csv.DictReader(path.open(), delimiter='\t' if path.suffix == '.tsv' else ','))


def markdown_table(rows, fields):
    def fmt(value):
        if isinstance(value, float):
            return f'{value:.6f}' if math.isfinite(value) else 'NA'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(fields) + ' |',
             '| ' + ' | '.join('---' for _ in fields) + ' |']
    lines += ['| ' + ' | '.join(fmt(row.get(k, '')) for k in fields) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def copy_source():
    out = HERE / 'source'
    for base in ['mxfold2', 'scripts', 'tests']:
        for folder, dirs, files in os.walk(ROOT / base):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', 'build', '.venv'}]
            for name in files:
                p = Path(folder) / name
                if p.suffix in SOURCE_EXT or name == 'CMakeLists.txt':
                    target = out / p.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(p, target)
    for name in ['pyproject.toml', 'uv.lock', 'LICENSE', 'README.md']:
        p = ROOT / name
        if p.exists():
            shutil.copyfile(p, out / name)


def primary_tables():
    TABLES.mkdir(exist_ok=True)
    provenance = []
    def copy_table(src, name):
        p = ROOT / src
        if p.exists():
            data = p.read_bytes()
            target = TABLES / name
            target.write_bytes(data)
            provenance.append({'table': 'tables/' + name, 'source': src,
                               'source_sha256': sha(data), 'bytes': len(data)})
    copy_table('reports/progress-20261022/evidence/comparison_epoch100_replay00_to50.tsv',
               'synthetic_epoch100_55.tsv')
    copy_table('reports/progress-20261022/evidence/synthetic_all_curves.tsv', 'synthetic_all_curves.tsv')
    for short, folder in [('rank', 'imle-rank-b-adabelief-trainonly-rnagym-20261020'),
                          ('fy', 'shape-fy-b-adabelief-full-rnagym-20261018-r1'),
                          ('nll', 'imle-nll-b-adabelief-trainonly-rnagym-20261020-r2')]:
        for name in ['evaluation_curve.tsv', 'training_curve.tsv']:
            copy_table('qsub_logs/rnagym/' + folder + '/' + name, 'real_' + short + '_' + name)
    copy_table('qsub_logs/eternafold/eternafold-joint-20261022/summary.tsv', 'eternafold_joint.tsv')
    copy_table('qsub_logs/eternafold/eternafold-joint-20261022/evaluation/testsetb_similarity_filtered_metrics.tsv',
               'eternafold_no_hit_subset.tsv')
    copy_table('qsub_logs/rnagym/rnagym-rank-improvements-20261003-01/seed_summary.csv', 'short_budget_seed_summary.csv')
    copy_table('qsub_logs/rnagym/rnagym-b-mw-shapefy-20261011/summary.tsv', 'mw_fy_smallbudget.tsv')
    copy_table('qsub_logs/rnagym/partial-fy-coverage-20261012/summary.tsv', 'partial_fy_summary.tsv')
    copy_table('qsub_logs/rnagym/imle-mwu-retention-20261010-v2/pilot_summary.tsv', 'mw_retention.tsv')
    copy_table('qsub_logs/implicit_mle_epoch100_new_parameter_measured/summary_rank_shape_by_setting.csv',
               'measured_rank_fy_all_settings.csv')
    copy_table('qsub_logs/implicit_mle_epoch100_new_parameter_measured/summary_rank_shape_per_run.csv',
               'measured_rank_fy_per_run.csv')
    copy_table('.worktrees/ribonanzanet-frozen-1d/qsub_logs/ribonanzanet/frozen1d-20261007/evaluation/summary.csv',
               'ribonanzanet_frozen1d.csv')
    # Recompute NLL/pseudo and matched-source means from per-RNA result files.
    runs = []
    roots = [ROOT / 'qsub_logs/implicit_mle_epoch100_new_parameter_measured/nll_sweep',
             ROOT / 'qsub_logs/implicit_mle_epoch100_new_parameter_measured/pseudo_sweep',
             ROOT / 'qsub_logs/implicit_mle_rnet_vs_measured']
    for root in roots:
        for p in sorted(root.rglob('condition.txt')):
            cond = {}
            for line in p.read_text().splitlines():
                if '=' in line:
                    k, v = line.split('=', 1)
                    cond[k] = v
            run = p.parent
            for epoch in [1, 5]:
                f = run / f'eval/TestSetB/epoch-{epoch}/result.csv'
                r = run / f'eval/CD80-rank/epoch-{epoch}/result_rank.csv'
                if not f.exists():
                    continue
                rows = read_rows(f)
                if not rows or 'F' not in rows[0]:
                    continue
                rank_rows = read_rows(r) if r.exists() else []
                rhos = [float(x['rho_paired']) for x in rank_rows
                        if x.get('rho_paired', '').lower() not in ('', 'nan', 'na')]
                runs.append({**cond, 'run_path': str(run.relative_to(ROOT)), 'epoch': epoch,
                             'F1': statistics.mean(float(x['F']) for x in rows),
                             'structure_count': len(rows), 'rank_count': len(rhos),
                             'rho_paired': statistics.mean(rhos) if rhos else float('nan')})
    write_tsv(TABLES / 'recomputed_nll_pseudo_matched_per_run.tsv', runs)
    # Shared epoch-80 joint table, plus actual last frozen epoch per condition.
    joint80, joint_latest = [], []
    base = ROOT / 'qsub_logs/rnagym/mxfold2-trainsetab-joint-20261022/conditions'
    for f in sorted(base.glob('*/evaluation_curve.tsv')):
        rows = read_rows(f)
        for row in rows:
            if int(row['epoch']) == 80:
                joint80.append({'condition': f.parent.name, **row})
        if rows:
            joint_latest.append({'condition': f.parent.name, **rows[-1]})
        copy_table(str(f.relative_to(ROOT)), 'joint_curve_' + f.parent.name + '.tsv')
    write_tsv(TABLES / 'mxfold2_joint_epoch80.tsv', joint80)
    write_tsv(TABLES / 'mxfold2_joint_latest.tsv', joint_latest)
    # Final values for longer adaptation diagnostics; preserve full curves too.
    long_rows = []
    for name in ['imle-rank-b-adabelief-state-3k100-20261014c',
                 'imle-rank-b-adabelief-fresh-control-3k100-20261014',
                 'imle-rank-b-noreplay-thermo-3k100-20261013',
                 'imle-rank-b-replay-thermo-3k100-20261012',
                 'imle-rank-thermo-3k100-20261012', 'imle-rank-thermo-pilot-20261012']:
        p = ROOT / 'qsub_logs/rnagym' / name / 'summary.tsv'
        if not p.exists():
            continue
        groups = {}
        for row in read_rows(p):
            groups.setdefault(row.get('condition', ''), []).append(row)
        for key, rows in groups.items():
            long_rows.append({'workflow': name, **rows[-1]})
        copy_table(str(p.relative_to(ROOT)), 'long_' + name + '.tsv')
    write_tsv(TABLES / 'long_adaptation_final.tsv', long_rows)
    write_tsv(TABLES / 'PROVENANCE.tsv', provenance)


def historical_reports():
    files = sorted((ROOT / 'scripts').glob('**/*.md'))
    files += [ROOT / 'reports/progress-20261022/README.md',
              ROOT / 'reports/progress-20261022/speaker_notes.md',
              ROOT / 'reports/eternafold-feasibility-20261022/README.md',
              ROOT / '.worktrees/ribonanzanet-frozen-1d/scripts/ribonanzanet/README.md',
              ROOT / 'qsub_logs/rnagym/replay20-full-test-20261023/README.md']
    lines = ['# 既存設計書・検証報告の全文保存\n',
             '各文書は元の記述を保持する。実行中/最新/現在という語は原文の時点に依存する。'
             '採否・新しい結果・評価不具合はPAPER_HANDOFF_ja.mdを優先する。\n']
    for f in files:
        if f.exists():
            data = f.read_bytes()
            lines += ['\n---\n', '元ファイル: `' + str(f.relative_to(ROOT)) + '`\n',
                      'SHA256: `' + sha(data) + '`\n', data.decode()]
    (HERE / 'HISTORICAL_REPORTS_ja.md').write_text('\n'.join(lines))


def catalog_and_appendix():
    records = []
    with zipfile.ZipFile(HERE / 'evidence.zip') as z:
        names = z.namelist()
        for name in names:
            parts = Path(name).parts
            if len(parts) < 3 or parts[0] != 'qsub_logs':
                continue
            rootlen = 3 if parts[1] in {'rnagym', 'eternafold', 'efold_hf'} else 2
            root = '/'.join(parts[:rootlen])
            if any(tag in Path(name).name.lower() for tag in
                   ['summary', 'plan', 'complete', 'results', 'preflight', 'selection', 'handoff', 'status']):
                records.append({'workflow': root, 'artifact': name,
                                'bytes': z.getinfo(name).file_size})
    write_tsv(HERE / 'EXPERIMENT_CATALOG.tsv', records, ['workflow', 'artifact', 'bytes'])
    lines = ['# 数値結果付録\n',
             '取得日時: ' + datetime.now(ZoneInfo('Asia/Tokyo')).isoformat() + '\n',
             '原表はtables/へ保存。全原ファイルの索引はEVIDENCE_INDEX.tsv、'
             'その他の系列・失敗記録はEXPERIMENT_CATALOG.tsvとevidence.zipを参照。'
             '表示は原則6桁へ丸め、原表は元精度を保持する。\n']
    sections = [
        ('合成55条件・固定100epoch', 'synthetic_epoch100_55.tsv',
         ['name', 'TestSetB_PPV', 'TestSetB_SEN', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman', 'TrainSetB_training_F1']),
        ('実測Rank・全評価epoch', 'real_rank_evaluation_curve.tsv', None),
        ('実測SHAPE-FY・全評価epoch', 'real_fy_evaluation_curve.tsv', None),
        ('実測NLL・全評価epoch', 'real_nll_evaluation_curve.tsv', None),
        ('EternaFold joint最終', 'eternafold_joint.tsv', None),
        ('EternaFold相同性除外の内訳', 'eternafold_no_hit_subset.tsv', None),
        ('MXfold2 joint共通80epoch（途中結果）', 'mxfold2_joint_epoch80.tsv',
         ['condition', 'epoch', 'TestSetB_PPV', 'TestSetB_SEN', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman']),
        ('MXfold2 joint固定100epoch（最終・全12条件）', 'mxfold2_joint_epoch100.tsv',
         ['name', 'epoch', 'TestSetB_PPV', 'TestSetB_SEN', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman']),
        ('MXfold2 joint：取得時の実際の最新評価', 'mxfold2_joint_latest.tsv',
         ['condition', 'epoch', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman']),
        ('RNAgym全433,539配列：replay10/20・固定e3主比較', 'rnagym_replay10_20_fulltest_macro.tsv',
         ['method','replay_percent','epoch','f1','auc','mae','mcc','TestSetB_F1']),
        ('RNAgym全test：probe別とAll_pooledを含む全28行', 'rnagym_replay10_20_fulltest.tsv',
         ['model','modifier','f1','auc','mae','mcc','TestSetB_F1']),
        ('RNAgym全test：coverageとcheckpoint監査', 'rnagym_fulltest_coverage.tsv',
         ['model','probe','unique_test_sequences','probe_sequences','observed_bases','shard_markers','prediction_archives']),
        ('eFold公開データ：固定endpointの全指標', 'efold_fixed_endpoint.tsv',
         ['arm','epoch','dataset','n','precision','recall','f1','tp','fp','fn','micro_f1']),
        ('eFold公開データ：全epoch・全84行', 'efold_all_epochs.csv', None),
        ('eFold公開データ：化学相関と有効数・全42行', 'efold_chemical_spearman.tsv', None),
        ('eFold公開データ：空予測・参照・BPP異常監査', 'efold_prediction_audit.tsv',
         ['arm','epoch','dataset','n','predicted_empty_sequences','empty_reference_sequences','BPP_error_sequences','F1_recomputed']),
        ('eFold公開データ：学習完了・露出・更新数', 'efold_training_endpoints.tsv',
         ['arm','epoch','updates','primary_examples','replay_examples','mean_loss']),
        ('SN0 measured Rank/FY：全96設定', 'measured_rank_fy_all_settings.csv', None),
        ('SN0 measured NLL：全32設定（e5を原RNA表から再集計）', 'measured_nll_summary.tsv', None),
        ('実測/RNet matched：全8arm（e5を原RNA表から再集計）', 'matched_source_summary.tsv', None),
        ('短期B・選択後3seed集計', 'short_budget_seed_summary.csv',
         ['condition', 'n_seeds', 'TestSetB_F1_mean', 'TestSetB_F1_sample_sd', 'selected_epochs', 'difference', 'ci95_low', 'ci95_high']),
        ('MW / FY短期B：選択と固定最終', 'mw_fy_smallbudget.tsv',
         ['condition', 'phase', 'seeds', 'F1_mean', 'F1_sd', 'F1_delta_baseline']),
        ('partial FY coverage', 'partial_fy_summary.tsv',
         ['condition', 'budget', 'phase', 'seeds', 'F1_mean', 'F1_sd', 'F1_delta_baseline']),
        ('MW retention', 'mw_retention.tsv',
         ['condition', 'phase', 'n_seeds', 'f1_mean', 'f1_sd', 'difference_vs_matched_baseline']),
        ('長期適応の最終値', 'long_adaptation_final.tsv',
         ['workflow', 'condition', 'epoch', 'optimizer_step', 'TestSetB_f1', 'heldout_auc']),
        ('凍結RibonanzaNet転移', 'ribonanzanet_frozen1d.csv', None),
    ]
    for title, name, fields in sections:
        p = TABLES / name
        if not p.exists():
            lines += ['\n## ' + title + '\n', '元表が取得できなかった。数値を補完していない。\n']
            continue
        rows = read_rows(p)
        fields = fields or (list(rows[0]) if rows else [])
        fields = [f for f in fields if rows and f in rows[0]]
        numeric = []
        for row in rows:
            r = dict(row)
            for k in fields:
                try:
                    if any(c in r[k] for c in '.eE'):
                        r[k] = float(r[k])
                except (ValueError, TypeError):
                    pass
            numeric.append(r)
        lines += ['\n## ' + title + '\n', '原表: `tables/' + name + '`\n', markdown_table(numeric, fields)]
    lines += ['\n## その他の実験系列\n',
              '以下のrootに対応する全計画・集計・完了/中断証拠のpathはEXPERIMENT_CATALOG.tsvにある。'
              '旧BPP指標、別起点、探索値も保存するが現行主比較へ混ぜない。\n']
    roots = sorted(set(r['workflow'] for r in records))
    lines += ['- `' + r + '`' for r in roots]
    (HERE / 'RESULTS_APPENDIX_ja.md').write_text('\n'.join(lines) + '\n')


def supplements():
    idx = list(csv.DictReader((HERE / 'EVIDENCE_INDEX.tsv').open(), delimiter='\t'))
    extra = []
    worktree = ROOT / '.worktrees/ribonanzanet-frozen-1d'
    for base in ['mxfold2', 'scripts/ribonanzanet', 'tests']:
        for folder, dirs, files in os.walk(worktree / base):
            dirs[:] = [d for d in dirs if d not in {'__pycache__', '.venv', 'build'}]
            for name in files:
                p = Path(folder) / name
                if p.suffix in SOURCE_EXT or name in {'LICENSE', 'CMakeLists.txt'}:
                    extra.append((p, str(p.relative_to(ROOT))))
    run = worktree / 'qsub_logs/ribonanzanet/frozen1d-20261007'
    for p in run.glob('*'):
        if p.is_file() and p.suffix in {'.json', '.csv', '.tsv', '.conf', '.yaml', '.sh', '.log'}:
            extra.append((p, str(p.relative_to(ROOT))))
    for sub in ['evaluation', 'baseline_verification']:
        extra += [(p, str(p.relative_to(ROOT))) for p in (run / sub).glob('*')
                  if p.is_file() and p.suffix in {'.json', '.csv', '.tsv', '.txt'}]
    baseline = ROOT / 'qsub_logs/rnagym/trainsetb-synthetic-20261020/initial_baseline.pth'
    extra.append((baseline, 'portable_data/initial_baseline.pth'))
    for workflow in (ROOT / 'qsub_logs/eternafold').iterdir():
        for p in workflow.glob('runs/*/model.params'):
            extra.append((p, str(p.relative_to(ROOT))))
    packed = ROOT / 'qsub_logs/rnagym/rnagym-full-coverage-20261003-01/data'
    for name in ['test_panel.parquet', 'test_manifest.json', 'manifest.json']:
        extra.append((packed / name, 'portable_data/rnagym_panel/' + name))
    for probe in ['2A3', 'DMS']:
        folder = ROOT / 'qsub_logs/rnagym/trainsetb-synthetic-20261020/data' / probe
        extra += [(p, 'portable_data/synthetic/' + probe + '/' + p.name)
                  for p in folder.iterdir() if p.is_file() and p.stat().st_size < 2 * 1024 * 1024]
    for label, base in [('measured', '/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data'),
                        ('rnet_matched', '/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_rnet_matched_shape_data')]:
        folder = Path(base)
        extra += [(p, 'external_inputs/' + label + '/' + p.name)
                  for p in folder.iterdir() if p.is_file() and p.stat().st_size < 4 * 1024 * 1024]
        if label == 'rnet_matched':
            extra += [(p, 'external_inputs/rnet_matched/rnet_core/' + p.name)
                      for p in (folder / 'rnet_core').iterdir() if p.is_file()]
    # Actual SN0 training/evaluation profiles used by either ID selection.
    folder = Path('/gs/bs/tga-satolab-gtex/yamauchi/data/Ribonanza/SN0/SHAPE')
    ids = set()
    for label in ['measured', 'rnet_matched']:
        path = Path('/gs/bs/tga-satolab-casp/satoken/tmp') / (
            'mxfold2_measured_shape_data' if label == 'measured' else 'mxfold2_rnet_matched_shape_data') / 'common_ids.lst'
        ids.update(path.read_text().splitlines())
    ids.update(Path('/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data/cd80_ids.lst').read_text().splitlines())
    extra += [(folder / (name + '.shape.tsv'), 'external_inputs/SN0_SHAPE/' + name + '.shape.tsv') for name in sorted(ids)]
    with zipfile.ZipFile(HERE / 'evidence.zip', 'a', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        existing = set(z.namelist())
        for p, arc in extra:
            if arc in existing or not p.exists() or p.is_symlink():
                continue
            data = p.read_bytes()
            z.writestr(arc, data)
            idx.append({'path': arc, 'original_absolute_path': str(p),
                        'bytes': len(data), 'sha256': sha(data), 'mtime_ns': p.stat().st_mtime_ns})
            existing.add(arc)
        indexed = {r['path'] for r in idx}
        for name in z.namelist():
            if name not in indexed:
                data = z.read(name)
                idx.append({'path': name, 'bytes': len(data), 'sha256': sha(data)})
    write_tsv(HERE / 'EVIDENCE_INDEX.tsv', idx,
              ['path', 'original_absolute_path', 'bytes', 'sha256', 'mtime_ns'])
    manifest = json.loads((HERE / 'CAPTURE_MANIFEST.json').read_text())
    manifest.update(supplement_captured_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                    evidence_files=len(idx), evidence_zip_bytes=(HERE / 'evidence.zip').stat().st_size,
                    evidence_uncompressed_bytes=sum(int(r['bytes']) for r in idx),
                    supplemental_contents='Frozen RibonanzaNet source and results; baseline with optimizer; all three structural datasets; synthetic packed arrays; RNAgym test panel; SN0 selected train/evaluation profiles and matched RNet cores. Large official chemical training arrays and other checkpoints remain external.')
    (HERE / 'CAPTURE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    copy_source()
    primary_tables()
    historical_reports()
    supplements()
    catalog_and_appendix()
    print('Tables, narrative appendices, source and supplemental evidence assembled.')
