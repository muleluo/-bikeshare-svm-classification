#!/usr/bin/env python3
"""
调整Word文档中的图片尺寸
"""
from docx import Document
from docx.shared import Inches

doc = Document("实验报告.docx")

max_width = Inches(5.5)
max_height = Inches(8.0)

print("正在调整图片尺寸...")
for i, shape in enumerate(doc.inline_shapes, 1):
    original_width = shape.width
    original_height = shape.height
    aspect_ratio = original_height / original_width

    is_flowchart = aspect_ratio > 1.2

    if is_flowchart:
        new_width = max_width
        new_height = int(max_width.emu * aspect_ratio)

        if new_height > max_height.emu:
            new_height = max_height.emu
            new_width = int(max_height.emu / aspect_ratio)
            print(f"  ✓ 图片 {i}: 流程图 -> {new_width/914400:.1f}×{new_height/914400:.1f}英寸 (高度受限)")
        else:
            print(f"  ✓ 图片 {i}: 流程图 -> {new_width/914400:.1f}×{new_height/914400:.1f}英寸")

        shape.width = int(new_width)
        shape.height = int(new_height)
    else:
        print(f"  - 图片 {i}: 数据图 {original_width/914400:.1f}×{original_height/914400:.1f}英寸")

doc.save("实验报告.docx")
print("\n✅ 已保存")

# 验证
print("\n=== 验证结果 ===")
doc2 = Document("实验报告.docx")
for i, shape in enumerate(doc2.inline_shapes, 1):
    w = shape.width / 914400
    h = shape.height / 914400
    status = "✅" if (w <= 6.0 and h <= 8.5) else "⚠️"
    print(f"{status} 图片 {i}: {w:.1f}×{h:.1f}英寸")
