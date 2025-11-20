#!/usr/bin/env python3
"""
创建HTML格式的可视化报告
"""

import os
from datetime import datetime

def create_html_report(results_dir='results', output_file='results/report.html'):
    """创建HTML报告"""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RNA-蛋白质相互作用预测 - 训练和评估报告</title>
    <style>
        body {{
            font-family: 'Arial', 'Microsoft YaHei', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 5px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .image-title {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 18px;
        }}
        .info {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RNA-蛋白质相互作用预测模型</h1>
        <h2>训练和评估报告</h2>
        
        <div class="info">
            <strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>报告目录:</strong> {results_dir}
        </div>
        
        <h2>1. 训练曲线</h2>
        <div class="image-container">
            <div class="image-title">训练和验证损失曲线</div>
            <img src="training_curves.png" alt="训练曲线">
        </div>
        
        <h2>2. 模型评估指标</h2>
        <div class="image-container">
            <div class="image-title">评估指标总结</div>
            <img src="metrics_summary.png" alt="指标总结">
        </div>
        
        <h2>3. 混淆矩阵</h2>
        <div class="image-container">
            <div class="image-title">分类混淆矩阵</div>
            <img src="confusion_matrix.png" alt="混淆矩阵">
        </div>
        
        <h2>4. ROC曲线</h2>
        <div class="image-container">
            <div class="image-title">接收者操作特征曲线 (ROC)</div>
            <img src="roc_curve.png" alt="ROC曲线">
        </div>
        
        <h2>5. PR曲线</h2>
        <div class="image-container">
            <div class="image-title">精确率-召回率曲线 (Precision-Recall)</div>
            <img src="pr_curve.png" alt="PR曲线">
        </div>
        
        <div class="footer">
            <p>报告由 RNA-蛋白质相互作用预测项目自动生成</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML报告已保存到: {output_file}")
    print(f"   可以在浏览器中打开查看: file://{os.path.abspath(output_file)}")

if __name__ == '__main__':
    create_html_report()
