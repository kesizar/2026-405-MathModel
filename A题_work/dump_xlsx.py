# -*- coding: utf-8 -*-
import openpyxl, os

BASE = r'D:/HuaweiMoveData/Users/17669/Desktop/A 题'
OUT = r'D:/A题_work'

for name in ['附件1', '附件2', 'result1', 'result2', 'result3']:
    path = os.path.join(BASE, name + '.xlsx')
    wb = openpyxl.load_workbook(path, data_only=True)
    lines = []
    lines.append('########## FILE: %s ##########' % name)
    for ws in wb.worksheets:
        lines.append('==== SHEET: %s (dims %s) ====' % (ws.title, ws.dimensions))
        for row in ws.iter_rows(values_only=True):
            # filter fully empty rows
            if all(c is None or str(c).strip() == '' for c in row):
                continue
            lines.append(' | '.join('' if c is None else str(c) for c in row))
    text = '\n'.join(lines)
    with open(os.path.join(OUT, name + '_dump.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    print(name, 'rows written, len', len(text))
print('DONE')
