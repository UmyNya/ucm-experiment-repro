#!/usr/bin/env python3
"""Generate a local Chinese Markdown report from explicitly selected run directories."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
from urllib.parse import quote

from common import new_run, read_json, save_json, sha256, utc
from report_data import collect, aggregate
from report_plots import render


def relative(path, report):
    return quote(os.path.relpath(Path(path).resolve(), report), safe='/.-_')


def number(value):
    return 'N/A' if value is None else f'{value:,.3f}' if isinstance(value, float) else str(value)


def generate(runs, *, include_validation=False, output_root=None, formats=None):
    runs = [Path(p).resolve() for p in runs]
    if len(runs) != len(set(runs)):
        raise ValueError('A run directory was selected more than once; refusing double-counted statistics')
    records = [row for run in runs for row in collect(run)]
    if not records:
        raise ValueError('Selected directories contain no attempts; startup-only records are not performance runs')
    config = copy.deepcopy(records[0]['plan']['configuration'])
    config['configuration_sources'] = []  # Source runs contain immutable copies; do not re-read changed live configs.
    formats = tuple(formats or config['profiling']['output_formats'])
    if output_root:
        config['paths']['output_root'] = str(Path(output_root).resolve())
    report = new_run(config, 'selected_runs', 'report')
    groups = aggregate(records, include_validation)
    data = {'schema_version': 1, 'generated_utc': utc(), 'selected_runs': [str(Path(p).resolve()) for p in runs],
            'validation_data_explicitly_included': include_validation, 'groups': groups,
            'records': [{k:v for k,v in r.items() if k not in {'samples', 'plan'}} for r in records]}
    save_json(report/'RESULTS.json', data)
    save_json(report/'ALL_CONFIGURATIONS.json', {r['path']: r['plan'] for r in records})
    with (report/'performance.csv').open('w', newline='') as stream:
        columns = ['run','attempt','phase','label','valid','synthetic','requests','successful_requests','duration_s','requests_per_s','input_tokens_per_s','output_tokens_per_s','ttft_mean_ms','ttft_p95_ms','e2e_p99_ms']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            metrics = row['performance'] or {}
            writer.writerow({'run': row['run'], 'attempt': row['path'], **{k:row[k] for k in ['phase','label','valid','synthetic']},
                             **{k:metrics.get(k) for k in columns[6:12]},
                             'ttft_mean_ms': (metrics.get('ttft_ms') or {}).get('mean'),
                             'ttft_p95_ms': (metrics.get('ttft_ms') or {}).get('p95'),
                             'e2e_p99_ms': (metrics.get('e2e_ms') or {}).get('p99')})
    charts = render(records, groups, report/'figures', formats)
    save_json(report/'figures.json', charts)
    lines = ['# 单配置实验记录汇总', '', f'生成时间：{data["generated_utc"]}。只汇总命令中明确选择的运行目录。', '',
             '[结构化结果](RESULTS.json) · [逐轮CSV](performance.csv) · [所有实际计划与配置](ALL_CONFIGURATIONS.json)', '',
             '输入指纹、模型配置、后端、输出设置、profiling 与服务管理范围不同的记录分别分组。图表使用 G 编号对应下表；同一配置的重复轮按完整参数指纹聚合，不按成绩挑选。', '']
    if any(r['synthetic'] for r in records):
        lines += ['**包含 CPU 模拟验证记录。其吞吐和延迟是合成数据，不能用于推理性能结论。** '+('已按显式参数单独绘图。' if include_validation else '默认不纳入指标聚合或图表。'), '']
    lines += ['## 每轮状态', '', '| 标签 | 阶段 | 结果 | 原始数据 | 排除原因 |', '|---|---|---|---|---|']
    for row in records:
        reasons = '; '.join(row['excluded_reasons']) or ('模拟数据，默认排除' if row['synthetic'] and not include_validation else '—')
        lines.append(f'| {row["label"]} | {row["phase"]} | {"有效" if row["valid"] else "排除"} | [记录]({relative(row["path"], report)}) | {reasons} |')
    lines += ['', '## 聚合性能', '', '轮间吞吐/耗时给出算术平均和样本标准差；只有一轮时标准差为 N/A。延迟分位数来自成功请求合并样本，不能把各轮 P95 简单平均当成总体 P95。', '',
              '| 组 | 标签 | 有效轮 | 请求数 | 输入tokens/s 平均 ± SD | 请求/s | 耗时/s | TTFT平均/ms | TTFT P50/P95/P99/ms | E2E平均/P95/P99/ms |',
              '|---|---|---:|---:|---|---:|---:|---:|---|---|']
    for group in groups:
        metric, pooled = group['metrics'], group['pooled']
        ttft, e2e = pooled['ttft_ms'] or {}, pooled['e2e_ms'] or {}
        lines.append('| '+' | '.join([group['id'], ', '.join(group['labels']), str(group['runs']), str(group['requests']),
                     number(metric['input_tokens_per_s']['mean'])+' ± '+number(metric['input_tokens_per_s']['sample_sd']),
                     number(metric['requests_per_s']['mean']), number(metric['duration_s']['mean']), number(ttft.get('mean')),
                     '/'.join(number(ttft.get(k)) for k in ['p50','p95','p99']), '/'.join(number(e2e.get(k)) for k in ['mean','p95','p99'])])+' |')
    lines += ['', '输入吞吐包含复用的前缀 tokens，不等于设备新增计算吞吐。输出仅 1 token 时 TPOT/ITL 不适用。profiling 轮有采集开销，与无 profiling 轮分组；独立客户端复用服务的结果不声明符合 fresh-service 冷缓存协议。单轮和非同期对照不支持统计显著性结论。', '']
    pipelines=[r for r in records if r['pipeline'] and r['valid'] and (not r['synthetic'] or include_validation)]
    if pipelines:
        lines += ['## Pipeline 指标', '', '| 标签/轮 | L2读 GiB | L2写 GiB | H2D/计算重叠 ms | 响应后物理尾部 ms |', '|---|---:|---:|---:|---:|']
        for row in pipelines:
            s=row['pipeline'];totals=s['capture_totals']
            lines.append('| '+' | '.join([row['label']+'/'+Path(row['path']).parent.name,
                         number(totals['L2_read']['bytes']/2**30),number(totals['L2_write']['bytes']/2**30),
                         number(s['overlap_ms']['client']['H2D']['device_compute']),number(s['physical_tail_ms'])])+' |')
        lines += ['', '阶段 union 时间可以重叠，不能相加当总耗时。共享 batch 的设备区间会出现在多个请求图中，不能累加请求行。设备 compute 表示已关联 kernel 活动，不是 SM/AI Core 利用率；H2D 完成量不是 HBM 实时驻留量。没有本请求 L2 读的 external 命中可能来自其他请求的预取。设备 trace 未提供 copy 字节时保持 N/A，边界原生计数另存。READY 的原始 scheduler 时间戳可能位于同一轮 READY receipt 之前，报告同时提供 first execute 时刻。', '']
    lines += ['## 图表', '']
    for chart in charts:
        png = next((p for p in chart['files'] if p.endswith('.png')), chart['files'][0])
        lines += ['### '+chart['title'], '', f'![{chart["title"]}]({relative(png, report)})', '',
                  ' · '.join(f'[{Path(p).suffix[1:].upper()}]({relative(p, report)})' for p in chart['files']), '']
    lines += ['## 实际参数与验证证据', '']
    seen = set()
    files = {}
    for row in records:
        source, plan = Path(row['path']), row['plan']
        lines += [f'### {row["label"]} · {source.parent.name} · {source.name}', '',
                  f'输入指纹：`{row["input_fingerprint"]}`；模式：{plan["mode"]}；范围：{row["scope"]}。', '']
        if row.get('observed_hbm'):
            lines += ['自动 HBM 实测容量（各 worker 分别记录）：', '',
                      '| worker rank | KV 字节数 | KV 物理 tokens |', '|---|---:|---:|']
            for worker in row['observed_hbm']:
                lines.append(f'| {worker["rank"]} | {worker["kv_tensor_bytes"]} | {worker["hbm_physical_tokens"]} |')
            lines += ['', '自动规划出的实际容量不同的记录分别分组，不合并为同一配置的重复轮。', '']
        for name in ['compilation-check.json', 'before-capacity-validation.json', 'after-capacity-validation.json',
                     'before-workers.json', 'after-workers.json', 'native-statistics.json', 'cache-preparation.json', 'service-metrics.jsonl',
                     'client-settings.json', 'client-command.json', 'pipeline-summary.json',
                     'pipeline-requests.json','pipeline-io.json','pipeline-diagnostics.json','pipeline-events.json.gz','profile-artifacts.json']:
            file = source/name
            if file.is_file():
                lines.append(f'- [{name}]({relative(file, report)})')
                files[str(file)] = sha256(file)
        service = source.parent/'service'
        artifacts = source/'profile-artifacts.json'
        if artifacts.is_file():
            capture = read_json(artifacts)
            for raw in [*capture.get('worker_directories',[]),*capture.get('device_traces',[]),capture.get('match_events')]:
                if raw and Path(raw).exists():
                    lines.append(f'- [采集原始数据：{Path(raw).name}]({relative(raw,report)})')
        for name in ['plan.json', 'startup-workers.json', 'runtime-inventory.json', 'service-environment.json', 'resource-cleanup.json', 'l2-storage.json']:
            file = service/name
            if file.is_file():
                lines.append(f'- [service/{name}]({relative(file, report)})')
                files[str(file)] = sha256(file)
        inputs=Path(row['run'])/'inputs'
        input_link=f'[输入快照]({relative(inputs,report)})' if inputs.exists() else '输入准备未完成'
        lines += ['', input_link+f' · [运行根目录]({relative(row["run"], report)})', '']
        key = row['configuration_group']
        if key not in seen:
            seen.add(key)
            lines += ['<details><summary>本组完整生效配置</summary>', '', '```json',
                      json.dumps(plan['configuration'], ensure_ascii=False, indent=2), '```', '', '</details>', '']
    lines += ['原始数据保留在统一运行目录中，以上链接均为相对路径。迁移或归档时同时保留报告与被引用的 runs 目录层级。完整服务 argv、环境覆盖、UCM JSON、模型身份、容量推导等均见 ALL_CONFIGURATIONS.json，不能把仅配置但未实测的值标成实测。', '',
              '## 范围与可信度', '', '已通过的客户端校验、容量观测和清理以各轮证据为准。缺少的观测保持 N/A；失败、校准和模拟数据不会自动成为正式成绩。目标 GPU/NPU 的兼容性只由实际运行证明；离线模拟验证本身不证明目标推理性能。', '']
    save_json(report/'evidence-files-sha256.json', files)
    (report/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    save_json(report/'status.json', {'status': 'generated', 'selected_records': len(records), 'valid_groups': len(groups),
              'figures': len(charts), 'report': 'REPORT.md'})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output-root', type=Path)
    p.add_argument('--include-validation', action='store_true', help='Explicitly include mock runs in separately marked validation figures')
    args = p.parse_args()
    print(generate(args.runs, include_validation=args.include_validation, output_root=args.output_root))


if __name__ == '__main__':
    main()
