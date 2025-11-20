#!/bin/bash
# 等待训练完成并自动评估

cd /Users/myt/pengtao/rna

echo "⏳ 等待训练完成..."
echo "   训练日志: logs/training.log"
echo ""

# 等待训练完成
while ! grep -q "Training completed!" logs/training.log 2>/dev/null; do
    sleep 30
    if [ -f logs/training.log ]; then
        last_epoch=$(grep "Epoch.*completed" logs/training.log | tail -1 | grep -oP "Epoch \K\d+" || echo "0")
        best_acc=$(grep "Best Val Acc" logs/training.log | tail -1 | grep -oP "Best Val Acc: \K[\d.]+" || echo "0")
        if [ "$last_epoch" != "0" ]; then
            echo "   当前进度: Epoch $last_epoch, 最佳验证准确率: ${best_acc}%"
        fi
    fi
done

echo ""
echo "✅ 训练完成！"
echo ""

# 显示最终结果
echo "=== 训练摘要 ==="
tail -5 logs/training.log | grep -E "(Epoch|Best Val Acc|Training completed)"
echo ""

# 运行评估
echo "📊 开始评估模型..."
python3 evaluate.py

echo ""
echo "✅ 评估完成！结果保存在 results/ 目录"

