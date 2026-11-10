"""Draw a publication-sized vector diagram with selectable text."""
from math import atan2, cos, sin, pi
from pathlib import Path
import reportlab
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
FONTS = Path(reportlab.__file__).parent / 'fonts'
pdfmetrics.registerFont(TTFont('Diagram', str(FONTS / 'Vera.ttf')))
pdfmetrics.registerFont(TTFont('DiagramBold', str(FONTS / 'VeraBd.ttf')))
c = canvas.Canvas(str(ROOT / 'architecture.pdf'), pagesize=(372, 178),
                  initialFontName='Diagram', initialFontSize=9.5)
c.setTitle('SemPointer: evidence selection and bounded descriptors')
c.setAuthor('Viksit Garg')

def box(x, y, w, h, title, lines):
    c.setFillGray(0.965)
    c.setStrokeGray(0.20)
    c.setLineWidth(0.7)
    c.roundRect(x, y, w, h, 3, fill=1, stroke=1)
    c.setFillGray(0.08)
    c.setFont('DiagramBold', 10)
    c.drawCentredString(x+w/2, y+h-15, title)
    c.setFont('Diagram', 9.5)
    for i, line in enumerate(lines):
        assert pdfmetrics.stringWidth(line, 'Diagram', 9.5) < w-10, line
        c.drawCentredString(x+w/2, y+h-29-i*12, line)

def arrow(points):
    c.setStrokeGray(0.25)
    c.setLineWidth(0.8)
    path = c.beginPath()
    path.moveTo(*points[0])
    for pt in points[1:]:
        path.lineTo(*pt)
    c.drawPath(path)
    x, y = points[-1]
    a = atan2(y-points[-2][1], x-points[-2][0])
    for direction in (-1, 1):
        b = a+pi+direction*.45
        c.line(x, y, x+5*cos(b), y+5*sin(b))

box(89, 130, 194, 46, 'Source registry + query',
    ['512-token blocks; stable identifiers'])
box(3, 60, 166, 58, 'Evidence selection',
    ['BM25 + coverage gate', 'Two or four text blocks'])
box(203, 60, 166, 58, 'Descriptor shortlist',
    ['Unselected top-16 blocks', '128-token allowance'])
box(46, 3, 204, 46, 'Complete chat input',
    ['Resolved text + descriptors; 1K/2K/4K'])
box(280, 3, 89, 46, 'Model', ['Answer'])
arrow([(145,130),(145,124),(86,124),(86,118)])
arrow([(227,130),(227,124),(286,124),(286,118)])
arrow([(86,60),(86,54),(120,54),(120,49)])
arrow([(286,60),(286,54),(210,54),(210,49)])
arrow([(250,26),(280,26)])
c.showPage()
c.save()
