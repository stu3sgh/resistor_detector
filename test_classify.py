#!/usr/bin/env python3
"""产线良品识别 - 分类测试脚本
用法: python3 test_classify.py [--scheme 1|2] [--region all|smd_components|main_chip|bottom_chip]
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

DATA_DIR = Path('/var/www/resistor/detection_results')
REGIONS = ['smd_components', 'main_chip', 'bottom_chip']


def test_scheme1(regions):
    """方案一: KNN 分类器"""
    from classifiers.knn_classifier import KNNClassifier
    results = []
    for region in regions:
        good_dir = DATA_DIR / region / 'good'
        bad_dir = DATA_DIR / region / 'bad'
        clf = KNNClassifier(str(DATA_DIR / region))
        
        for label_dir, label_name in [(good_dir, 'good'), (bad_dir, 'bad')]:
            if not label_dir.exists():
                continue
            for f in sorted(label_dir.glob('*.png')):
                r = clf.predict(str(f))
                results.append({
                    'region': region,
                    'file': f.name,
                    'label': label_name,
                    'result': r['result'],
                    'confidence': round(r['confidence'], 4)
                })
    return results


def test_scheme2(regions):
    """方案二: HOG+SVM V2 多区域分类器"""
    import cv2
    checker_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcb_resistor_checker')
    sys.path.insert(0, checker_dir)
    
    from infer_multi_region_hog_svm_v2 import (
        find_latest_versioned_model_dir, classify_direct_region,
        classify_smd_region
    )
    from detect_resistor_presence import load_config, load_template_state
    from roi_classifier_hog_svm import load_model_bundle
    
    models_root = Path(checker_dir) / 'models'
    config = load_config(Path(checker_dir) / 'config.yaml')
    template_state = load_template_state(config, Path(checker_dir) / 'config.yaml')
    
    smd_dir = find_latest_versioned_model_dir(models_root, 'hog_svm')
    main_dir = find_latest_versioned_model_dir(models_root, 'main_chip_hog_svm')
    bottom_dir = find_latest_versioned_model_dir(models_root, 'bottom_chip_hog_svm')
    
    smd_svm, smd_feat, smd_meta = load_model_bundle(smd_dir)
    main_svm, main_feat, main_meta = load_model_bundle(main_dir)
    bottom_svm, bottom_feat, bottom_meta = load_model_bundle(bottom_dir)
    
    results = []
    
    # SMD 区域（需要 ORB 配准）
    if 'smd_components' in regions:
        for label_name in ['good', 'bad']:
            label_dir = DATA_DIR / 'smd_components' / label_name
            if not label_dir.exists():
                continue
            for f in sorted(label_dir.glob('*.png')):
                t0 = time.time()
                r = classify_smd_region(f, config, template_state, smd_svm, smd_feat, smd_meta)
                t1 = time.time()
                results.append({
                    'region': 'smd_components',
                    'file': f.name,
                    'label': label_name,
                    'result': r['result'],
                    'reason': r.get('reason', ''),
                    'svm_margin': r.get('svm_margin_abs', 0),
                    'time_ms': round((t1 - t0) * 1000)
                })
    
    # main_chip / bottom_chip（直接 HOG+SVM）
    for region, svm, feat, meta in [
        ('main_chip', main_svm, main_feat, main_meta),
        ('bottom_chip', bottom_svm, bottom_feat, bottom_meta),
    ]:
        if region not in regions:
            continue
        for label_name in ['good', 'bad']:
            label_dir = DATA_DIR / region / label_name
            if not label_dir.exists():
                continue
            for f in sorted(label_dir.glob('*.png')):
                t0 = time.time()
                r = classify_direct_region(region, f, svm, feat, meta)
                t1 = time.time()
                results.append({
                    'region': region,
                    'file': f.name,
                    'label': label_name,
                    'result': r['result'],
                    'reason': r.get('reason', ''),
                    'svm_margin': r.get('svm_margin_abs', 0),
                    'time_ms': round((t1 - t0) * 1000)
                })
    
    return results


def print_results(results):
    total = len(results)
    correct = sum(1 for r in results if r['result'] == r['label'])
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f'\n{"="*60}')
    print(f'总样本: {total}  正确: {correct}  准确率: {accuracy:.1f}%')
    print(f'{"="*60}')
    
    for region in REGIONS:
        region_results = [r for r in results if r['region'] == region]
        if not region_results:
            continue
        rc = sum(1 for r in region_results if r['result'] == r['label'])
        ra = rc / len(region_results) * 100
        # 统计 TP/FP/FN/TN
        tp = sum(1 for r in region_results if r['label'] == 'good' and r['result'] == 'good')
        fp = sum(1 for r in region_results if r['label'] == 'bad' and r['result'] == 'good')
        fn = sum(1 for r in region_results if r['label'] == 'good' and r['result'] == 'bad')
        tn = sum(1 for r in region_results if r['label'] == 'bad' and r['result'] == 'bad')
        
        print(f'\n📊 {region} ({len(region_results)} 张, 准确率 {ra:.1f}%)')
        print(f'   TP={tp} FP={fp} FN={fn} TN={tn}')
        
        # 错误样本
        errors = [r for r in region_results if r['result'] != r['label']]
        if errors:
            print(f'   ❌ 误判:')
            for e in errors:
                margin = e.get('svm_margin', '')
                timing = e.get('time_ms', '')
                extra = f' margin={margin}' if margin else ''
                extra += f' {timing}ms' if timing else ''
                print(f'      {e["label"]}/{e["file"]} → 预测: {e["result"]}{extra}')


def main():
    parser = argparse.ArgumentParser(description='产线良品识别分类测试')
    parser.add_argument('--scheme', type=int, default=2, choices=[1, 2], help='分类方案 (1=KNN, 2=HOG+SVM)')
    parser.add_argument('--region', default='all', choices=['all', 'smd_components', 'main_chip', 'bottom_chip'],
                        help='测试区域')
    args = parser.parse_args()
    
    regions = REGIONS if args.region == 'all' else [args.region]
    
    print(f'🧪 测试方案 {args.scheme} | 区域: {", ".join(regions)}')
    
    t0 = time.time()
    if args.scheme == 1:
        results = test_scheme1(regions)
    else:
        results = test_scheme2(regions)
    elapsed = time.time() - t0
    
    print_results(results)
    print(f'\n⏱️ 总耗时: {elapsed:.1f}s')
    
    # 保存 JSON
    out = '/var/www/resistor/test_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'📄 详细结果: {out}')


if __name__ == '__main__':
    main()
