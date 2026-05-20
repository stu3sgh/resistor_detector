#!/bin/bash
# 同步 detection_results 数据到 GitHub（每15分钟执行）
REPO="/root/.openclaw/workspace-coder/portfolio/resistor/resistor_detector"
SRC="/var/www/resistor"
LOG="/var/www/resistor/sync.log"

source /root/.openclaw/.env

# 同步文件
cp "$SRC/server.py" "$REPO/"
cp "$SRC/index.html" "$REPO/"
rsync -a "$SRC/detection_results/" "$REPO/detection_results/"

cd "$REPO"

# 检查是否有变更
if [ -z "$(git status --short)" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') 无变更" >> "$LOG"
    exit 0
fi

CHANGES=$(git status --short | wc -l)
echo "$(date '+%Y-%m-%d %H:%M:%S') 发现 $CHANGES 个变更" >> "$LOG"

# 先 pull 远端变更，避免分叉
git pull --rebase https://${GITHUB_TOKEN}@github.com/stu3sgh/resistor_detector.git main >> "$LOG" 2>&1

git add -A
git commit -m "auto: 数据同步 $(date '+%m-%d %H:%M') - $CHANGES files" >> "$LOG" 2>&1

git push https://${GITHUB_TOKEN}@github.com/stu3sgh/resistor_detector.git main >> "$LOG" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S') 推送完成" >> "$LOG"
