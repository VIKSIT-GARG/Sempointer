"""Build the main SemPointer Word manuscript and its workflow figures."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "SemPointer.docx"
FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
if not Path(FONT).exists():
    FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size, bold=False):
    name = FONT.replace(".ttf", "-Bold.ttf") if bold else FONT
    if not Path(name).exists():
        name = FONT
    return ImageFont.truetype(name, size)


def box(draw, xy, title, subtitle="", fill="#eaf1f7", width=2):
    draw.rounded_rectangle(xy, radius=15, fill=fill, outline="#536b7d", width=width)
    x0, y0, x1, y1 = xy
    cx = (x0+x1)/2
    draw.text((cx, y0+17), title, font=font(29, True), fill="#152b3b", anchor="mt")
    if subtitle:
        draw.multiline_text((cx, y0+60), subtitle, font=font(22), fill="#33495a", anchor="mt", align="center", spacing=5)


def arrow(draw, start, end, color="#345a73", width=6):
    import math
    draw.line((start, end), fill=color, width=width)
    a = math.atan2(end[1]-start[1], end[0]-start[0])
    for da in (-0.43, 0.43):
        p = (end[0]-21*math.cos(a+da), end[1]-21*math.sin(a+da))
        draw.line((end, p), fill=color, width=width)


def workflow():
    im = Image.new("RGB", (1800, 930), "white")
    d = ImageDraw.Draw(im)
    d.text((75, 38), "INGEST ONCE", font=font(25, True), fill="#536b7d")
    d.text((75, 367), "FOR EACH QUESTION", font=font(25, True), fill="#536b7d")
    top = [(80,90,390,270), (510,90,820,270), (940,90,1250,270), (1370,90,1715,270)]
    box(d, top[0], "History", "documents or conversation")
    box(d, top[1], "Split into blocks", "up to 512 model tokens")
    box(d, top[2], "Pointer registry", "stable ID + up to 16 tokens", "#e3f3f6")
    box(d, top[3], "Text substrate", "ID maps to exact block text", "#e3f3f6")
    for a,b in zip(top,top[1:]): arrow(d,(a[2]+4,180),(b[0]-5,180))
    bot = [(80,430,350,620),(455,430,740,620),(850,430,1135,620),(1245,430,1715,620)]
    box(d,bot[0],"Question","query terms")
    box(d,bot[1],"BM25 rank","protect top two IDs")
    box(d,bot[2],"Coverage gate","two or up to four IDs","#fff0dd")
    box(d,bot[3],"Resolve and assemble","selected full text + other pointers","#e3f3f6")
    for a,b in zip(bot,bot[1:]): arrow(d,(a[2]+4,525),(b[0]-5,525))
    arrow(d,(1095,272),(1095,425))
    arrow(d,(1550,272),(1550,425))
    d.rounded_rectangle((345,740,1450,875),radius=15,fill="#f1f5f8",outline="#536b7d",width=2)
    d.text((897,765),"Generator answers from the mixed prompt",font=font(32,True),fill="#152b3b",anchor="mt")
    d.text((897,818),"Log route, selected IDs, tokens, score and latency",font=font(24),fill="#33495a",anchor="mt")
    arrow(d,(1480,624),(1310,735))
    d.multiline_text((760,645),"Expand if weighted coverage < 0.80, or\n< 0.95 with boundary score ratio >= 0.85",font=font(21),fill="#805322",align="center",spacing=3)
    im.save(ASSETS/"architecture_workflow.png",dpi=(240,240))


def chart():
    im=Image.new("RGB",(1600,720),"white");d=ImageDraw.Draw(im)
    blue="#166a9a"; orange="#c66a32"; gray="#6e7780"
    labels=["RULER","LongBench","LoCoMo"]
    a=[(62.1,2201,1252),(10.4,1878,1732),(6.9,1896,1849)]
    b=[(62.3,2416,1485),(10.6,2100,1867),(7.2,2122,1918)]
    d.text((45,20),"Observed quality and answer cost",font=font(34,True),fill="#152b3b")
    for i,label in enumerate(labels):
        y=105+i*195
        d.text((55,y),label,font=font(28,True),fill="#152b3b")
        d.text((55,y+47),"SemPointer",font=font(24),fill=blue)
        d.text((55,y+82),"BM25-4",font=font(24),fill=orange)
        d.text((365,y),"Score",font=font(23,True),fill="#152b3b")
        d.text((725,y),"Active tokens",font=font(23,True),fill="#152b3b")
        d.text((1135,y),"Answer ms",font=font(23,True),fill="#152b3b")
        for row,color,yy in ((a[i],blue,y+42),(b[i],orange,y+85)):
            d.text((365,yy),f"{row[0]:.1f}%",font=font(25),fill=color)
            d.rounded_rectangle((725,yy+4,725+int(row[1]/2500*350),yy+26),radius=8,fill=color)
            d.text((1135,yy),f"{row[2]:,}",font=font(25),fill=color)
    im.save(ASSETS/"quality_cost.png",dpi=(240,240))


def set_cell_shading(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement("w:shd");shd.set(qn("w:fill"),fill);tcPr.append(shd)


def set_borders(cell):
    tcPr=cell._tc.get_or_add_tcPr(); borders=OxmlElement("w:tcBorders")
    for edge in ("top","left","bottom","right"):
        el=OxmlElement("w:"+edge);el.set(qn("w:val"),"single");el.set(qn("w:sz"),"4");el.set(qn("w:color"),"D9D9D9");borders.append(el)
    tcPr.append(borders)


def table(doc,headers,rows,widths=None):
    if doc.paragraphs:
        doc.paragraphs[-1].paragraph_format.keep_with_next=True
    t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
    for j,h in enumerate(headers): t.rows[0].cells[j].text=h
    for row in rows:
        cells=t.add_row().cells
        for j,v in enumerate(row): cells[j].text=str(v)
    for ri,row in enumerate(t.rows):
        for j,c in enumerate(row.cells):
            if widths: c.width=Inches(widths[j])
            c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_borders(c)
            set_cell_shading(c,"DDE8F0" if ri==0 else ("F5F8FA" if ri%2==0 else "FFFFFF"))
            for p in c.paragraphs:
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
                if j>0:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name="Aptos";run.font.size=Pt(8.3)
                    if ri==0:run.bold=True
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t


def para(doc,text,boldlead=None):
    p=doc.add_paragraph(style="Normal")
    if boldlead:
        p.add_run(boldlead).bold=True
    p.add_run(text)
    return p


def caption(doc,text):
    p=doc.add_paragraph(style="Caption");p.paragraph_format.keep_with_next=False;p.add_run(text);return p


def figure(doc,path,width,cap):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
    p.add_run().add_picture(str(path),width=Inches(width))
    caption(doc,cap)


def build():
    workflow();chart()
    doc=Document();s=doc.sections[0];s.top_margin=Inches(.65);s.bottom_margin=Inches(.62);s.left_margin=Inches(.83);s.right_margin=Inches(.83)
    styles=doc.styles
    for name in ("Normal","Title","Heading 1","Heading 2","Caption"):
        st=styles[name];st.font.name="Aptos";st.font.color.rgb=RGBColor(0,0,0)
    styles["Normal"].font.size=Pt(12);styles["Normal"].paragraph_format.space_after=Pt(6);styles["Normal"].paragraph_format.line_spacing=1.18
    styles["Title"].font.size=Pt(17);styles["Title"].font.bold=True;styles["Title"].paragraph_format.space_after=Pt(12)
    title_ppr=styles["Title"]._element.get_or_add_pPr()
    for child in list(title_ppr):
        if child.tag==qn("w:pBdr"):
            title_ppr.remove(child)
    styles["Heading 1"].font.size=Pt(12);styles["Heading 1"].font.bold=True;styles["Heading 1"].paragraph_format.space_before=Pt(11);styles["Heading 1"].paragraph_format.space_after=Pt(5)
    styles["Heading 2"].font.size=Pt(10.5);styles["Heading 2"].font.bold=True;styles["Heading 2"].paragraph_format.space_before=Pt(8);styles["Heading 2"].paragraph_format.space_after=Pt(3)
    styles["Caption"].font.size=Pt(8.5);styles["Caption"].paragraph_format.space_after=Pt(8)
    doc.add_paragraph("SemPointer Adaptive Addressable Context for Long Context Question Answering",style="Title")
    author=doc.add_paragraph()
    author.alignment=WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_after=Pt(2)
    author.add_run("Viksit Garg").bold=True
    affiliation=doc.add_paragraph()
    affiliation.alignment=WD_ALIGN_PARAGRAPH.CENTER
    affiliation.paragraph_format.space_after=Pt(2)
    affiliation.add_run("Department of Artificial Intelligence and Machine Learning\nDr. Akhilesh Das Gupta Institute of Professional Studies, New Delhi, India")
    contact=doc.add_paragraph()
    contact.alignment=WD_ALIGN_PARAGRAPH.CENTER
    contact.paragraph_format.space_after=Pt(8)
    contact.add_run("viksitg427@gmail.com")
    for p in (author,affiliation,contact):
        p.paragraph_format.line_spacing=1.0
        for run in p.runs:
            run.font.name="Aptos"
            run.font.size=Pt(9)
    doc.add_heading("Abstract",1)
    para(doc,"Long context question answering must balance evidence coverage against the tokens and time used for every question. SemPointer stores history in indexed text blocks, keeps short address pointers in the active prompt, and resolves selected blocks for generation. Its lexical router protects the two highest ranked BM25 blocks and expands toward four when weighted query coverage or a close ranking boundary suggests that more evidence is needed. Across 180 selected RULER examples, SemPointer scores 62.1% against 62.3% for four block BM25 retrieval, while using 8.9% fewer active tokens and 15.7% less mean answer latency after ingestion. On selected LongBench and LoCoMo subsets, observed score differences are small and inconclusive, with lower mean tokens and latency. Relative to the prior attention fused implementation, the RULER score rises by 16.6 percentage points. These results support SemPointer as an inspectable context budget mechanism in this single model evaluation. They do not establish pointer specific causal value, accuracy equivalence, or official benchmark standing.")
    para(doc,"Keywords: long context language models; adaptive retrieval; address pointers; context efficiency")
    doc.add_heading("1 Introduction",1)
    para(doc,"Repeated questions about a large document or conversation require different evidence from the same stored history. Sending every token to the language model gives broad access but repeats a large prefill. Fixed passage retrieval reduces this cost, although a generous block count pays for evidence many queries may not need. SemPointer exposes a stable address for every block and uses a query dependent budget to choose which addresses become full text.")
    para(doc,"The pointer is a readable registry address, not a learned reconstruction code. The generation prompt contains full text for resolved blocks and compact pointers for unresolved blocks. This design makes evidence selection explicit and auditable: the system can log selected identifiers, the reason for expansion, prompt tokens, and answer timing for each question. The present experiments test the whole serving policy. They cannot yet attribute improvements to pointer content apart from the router.")
    para(doc,"An earlier attention fused selector combined BM25, dense similarity and a model attention pass. Its completed RULER evaluation scored 45.6%, versus 62.3% for four block BM25 retrieval. That failure motivated the measured architecture: lexical ranking first, a protected two block route, and conditional expansion. The principal result is a quality and cost point between fixed two block and four block retrieval, with scores near the latter on the evaluated subsets and lower mean answer cost.")
    doc.add_heading("2 Architecture and Workflow",1)
    figure(doc,ASSETS/"architecture_workflow.png",6.8,"Figure 1. SemPointer workflow. History is indexed once; each question ranks block IDs, chooses a two or expanded route, resolves selected text, and sends the mixed prompt to the generator. The tested path resolves text rather than cached key value states.")
    doc.add_heading("2.1 Ingestion and address registry",2)
    para(doc,"The ingester divides the source history into text blocks of at most 512 target model tokens. For each block i it stores the exact payload X_i and a stable, readable pointer P_i capped at 16 tokens. Pointer terms favor distinctive words, numbers and entity like strings. The registry holds (identifier, pointer, payload) bindings. An identifier must continue to resolve to the same payload throughout an evaluation so that logged evidence and answers can be inspected.")
    para(doc,"The active prompt contains the question, instructions, selected full blocks, and compact pointers for other indexed blocks. Its approximate length is the sum of unresolved pointer lengths, resolved block lengths, question length and framing overhead. Measured token counts include labels, instructions and the model chat template. Unresolved pointers consume tokens, so a comparison with the same adaptive router but no pointers is needed to test their separate contribution.")
    doc.add_heading("2.2 Query routing and resolution",2)
    para(doc,"BM25 ranks every indexed block for a question and protects the top two identifiers. The router computes coverage from distinct question terms found in those blocks, weighting each term by nonnegative inverse document frequency. Coverage is zero when the denominator is zero. It requests four candidate blocks if weighted coverage is below 0.80. It also expands when coverage is below 0.95 and the third BM25 score is at least 0.85 times the second. Block order breaks ties. A 2,560 token cap on resolved payload can reject later candidates, so the final number of full blocks can be below the requested count.")
    para(doc,"The resolver replaces selected addresses with exact payload text and leaves other addresses as compact entries. The generator answers from this mixed prompt. The route, selected IDs, coverage values, resolved token count and selection time are recorded per example. The measured configuration does not load a dense embedding model or use semantic rescue. Surface term coverage is only a heuristic; paraphrase and multi step evidence can make it overconfident.")
    doc.add_heading("2.3 Budget and route semantics",2)
    para(doc,"The active prompt can be understood as L(q) ≈ Σ(i outside S) |P_i| + Σ(i inside S) |X_i| + |q| + L_frame, where S is the resolved set, P_i is a compact pointer and X_i is exact block text. This expression is an accounting model rather than the tokenizer count: block labels, instructions and the chat template are included in the measured active tokens. Since each unresolved pointer remains in the prompt, the pointer index has a cost even when only two payloads are selected.")
    para(doc,"The coverage gate is applied after BM25 ranking, so it does not choose a new ranking function. The protected two blocks prevent the expansion cap from removing the strongest lexical candidates. The close boundary rule tests whether the third ranked block is competitive with the second. It is therefore a resource allocation heuristic, not an uncertainty estimate calibrated to answer correctness. A low coverage query can still fail if the answer is in a paraphrased block that BM25 ranks poorly; a high coverage query can still require several separate facts.")
    doc.add_page_break()
    caption(doc,"Table 1. Query route decision used in the completed runs. The payload cap can stop expansion before all requested candidates are resolved.")
    table(doc,["Condition after top two blocks","Requested route","Reason"],[
        ["Weighted coverage < 0.80","Up to four blocks","Question terms lack support"],
        ["Coverage < 0.95 and third score / second score ≥ 0.85","Up to four blocks","Ranking boundary is close"],
        ["Otherwise","Two blocks","Protected lexical evidence is retained"],
        ["Resolved payload exceeds 2,560 tokens","Stop adding blocks","Enforce text budget"]],[3.0,1.2,2.3])
    doc.add_heading("2.4 Related systems and scope",2)
    para(doc,"BM25 provides a strong, inexpensive lexical ordering for evidence retrieval (Robertson and Zaragoza, 2009). Retrieval augmented generation then sends selected evidence to a language model (Lewis et al., 2020). SemPointer shares that retrieval setting, but makes the document store explicitly addressable: its short entries name blocks that can be resolved to exact source text. It does not learn a compressed latent memory and does not change the base model's parameters.")
    para(doc,"Landmark attention offers random access inside a modified transformer (Mohtashami and Jaggi, 2023). Prompt compression and cache management seek related context savings through other mechanisms. Our measured path does not use a cache reuse or compression kernel. The comparison that matters here is the same model and run with fixed BM25 passage counts. In particular, any token advantage over BM25-4 could come from choosing fewer full passages; it should not be ascribed to the pointer text without a matched control.")
    doc.add_heading("3 Experimental Protocol",1)
    para(doc,"Every measured system uses Qwen2.5 3B Instruct in NF4 precision on one 8 GB GPU. RULER contains 180 examples from six selected synthetic tasks, with ten seeded examples per task at 4k, 8k and 16k. LongBench contains the first 100 examples from each of six English question answering tasks, and LoCoMo contains conversations 0 to 2, totaling 497 examples. The model, tokenizer, instructions, seed and generation cap are fixed within each collection: 32 new tokens for RULER and 64 for the others. All scheduled system example pairs completed without recorded errors.")
    para(doc,"Same run comparisons use SemPointer and BM25 with four resolved blocks on all three collections; RULER also includes BM25 with two blocks. Earlier attention fused and full context RULER outputs are compared only on shared example IDs for scores. Their timing came from a separate execution and is not used as a speed comparison. RULER uses case insensitive answer containment. Selected LongBench tasks use normalized question answering token F1. LoCoMo uses the project's normalized token F1 adaptation rather than the official category specific scorer. The selected tasks, prompts and output caps are not an official benchmark submission.")
    para(doc,"We report mean scores with 95% example bootstrap intervals. Paired differences use identical example IDs and 10,000 bootstrap resamples with seed 42; score p values use two sided sign flip tests. Intervals describe variation over these examples, not over model families or random seeds. Answer latency starts at the answer call and includes selection, assembly and generation; ingestion is excluded. Time to first token is reported separately.")
    doc.add_heading("3.1 Comparison and measurement boundaries",2)
    para(doc,"The four block comparator resolves BM25's highest ranked passages for every question. The two block comparator is measured on RULER. Both use the same generation model and benchmark subset as SemPointer in each completed run. This controls the largest sources of within run timing variation, but it does not equate prompt lengths. A strict iso token comparison would give BM25 and SemPointer an identical prompt budget for each example and is still missing.")
    para(doc,"All reported percentages are averages over the specified example sets. Confidence intervals and sign flip tests are descriptive for those sets; no multiple collection or model level replication is claimed. The answer call includes tokenization and model generation as implemented by the runner. Ingestion, registry construction and any future cache warmup must be charged separately when comparing complete applications. This matters especially for a system that is intended to answer many questions about one stored history.")
    doc.add_heading("4 Results",1)
    caption(doc,"Table 2. Same run results. Scores are mean percentages with 95% example bootstrap confidence intervals. Tokens, answer latency and time to first token are per question means; ingestion is excluded.")
    table(doc,["Task","System","n","Score [95% CI]","Tokens","Answer ms","TTFT ms"],[
        ["RULER","SemPointer",180,"62.1 [55.5, 68.7]","2,201","1,252",594],["RULER","BM25-2",180,"59.7 [52.9, 66.6]","1,381","1,051",442],["RULER","BM25-4",180,"62.3 [55.7, 68.9]","2,416","1,485",849],
        ["LongBench","SemPointer",600,"10.4 [9.2, 11.7]","1,878","1,732",457],["LongBench","BM25-4",600,"10.6 [9.4, 11.8]","2,100","1,867",494],
        ["LoCoMo","SemPointer",497,"6.9 [6.1, 7.7]","1,896","1,849",476],["LoCoMo","BM25-4",497,"7.2 [6.4, 8.1]","2,122","1,918",519]], [1.0,1.0,.35,1.52,.7,.76,.7])
    figure(doc,ASSETS/"quality_cost.png",6.65,"Figure 2. Mean score, active prompt tokens and answer latency in the same completed benchmark runs. The chart shows means; Table 3 gives paired uncertainty.")
    doc.add_heading("4.1 Quality and answer cost",2)
    para(doc,"SemPointer scores 62.1% on RULER versus 62.3% for BM25 with four blocks. Its active prompt averages 2,201 rather than 2,416 tokens and its answer call averages 1,252 rather than 1,485 ms. The paired score difference is −0.22 percentage points, with a 95% interval from −3.00 to +2.67; this does not prove accuracy equivalence. Relative to two block BM25, SemPointer gains 2.39 points, with an interval from −0.11 to +5.11, while spending more tokens and time. It is an intermediate measured point rather than a universal winner.")
    para(doc,"On LongBench the measured scores are 10.4% and 10.6% for SemPointer and BM25-4, with 223 fewer active tokens and 135 ms lower paired mean answer latency. On LoCoMo the adapted F1 scores are 6.9% and 7.2%, with 226 fewer tokens and 68 ms lower latency. Both score difference intervals include zero. These low score floors restrict the claim to cost at the observed level of task performance.")
    caption(doc,"Table 3. Paired SemPointer minus BM25-4 effects on shared example IDs. Negative token and latency differences favor SemPointer. Intervals are 95% paired bootstrap confidence intervals.")
    table(doc,["Task","Score difference pp [CI]","p","Token difference [CI]","Answer ms difference [CI]"],[
        ["RULER","−0.22 [−3.00, +2.67]","0.943","−215 [−311, −117]","−233 [−267, −198]"],
        ["LongBench","−0.13 [−1.04, +0.79]","0.775","−223 [−277, −167]","−135 [−166, −104]"],
        ["LoCoMo","−0.37 [−0.85, +0.13]","0.139","−226 [−263, −188]","−68 [−98, −39]"]],[.9,1.55,.45,1.45,1.62])
    para(doc,"The earlier attention fused SemPointer implementation scored 45.6% on the same 180 RULER IDs. The present system is 16.56 points higher, paired interval [11.00, 22.56], p < 0.001. Full context scored 69.8% in that earlier execution; the present system remains 7.67 points lower, paired interval [−12.11, −3.44]. The larger improvement is consistent with removing the harmful attention fused selector, while the full context gap remains.")
    doc.add_heading("4.2 Router decisions and latency accounting",2)
    para(doc,"The two block route is used on 118 of 180 RULER questions, 421 of 600 LongBench questions and 409 of 497 LoCoMo questions. The rest request expansion, subject to the resolved text cap. This routing pattern explains how mean active tokens can sit below a fixed four block policy. Route conditioned score differences would be observational because the two groups have different question difficulty; they are not evidence that expansion causes a particular score change.")
    caption(doc,"Table 4. Observed two block routing frequency. Counts describe choices made by the gate, not the causal effect of the route on answer quality.")
    table(doc,["Task","Two block questions","Total","Share"],[
        ["RULER",118,180,"65.6%"],["LongBench",421,600,"70.2%"],["LoCoMo",409,497,"82.3%"]],[1.7,1.8,1.0,1.0])
    para(doc,"Time to first token is 594 versus 849 ms for SemPointer and BM25-4 on RULER, 457 versus 494 ms on LongBench, and 476 versus 519 ms on LoCoMo. Selection itself averages 2.2, 2.8 and 2.0 ms for SemPointer, versus 1.3, 0.2 and 0.2 ms for BM25-4. The router adds a small selection cost while the complete answer call is faster in the same runs. Fewer active tokens are a plausible contributor, but the measurements do not isolate kernel level causes. Every example is ingested separately in this runner, so these data do not establish many query amortization or serving concurrency performance.")
    doc.add_heading("4.3 Failure analysis",2)
    para(doc,"On the 30 example RULER multikey subset, SemPointer and BM25-4 both score 93.3%. One single needle variant is 86.7% versus 80.0% for BM25-4, while variable tracking is 28.7% versus 36.7%. These small task slices do not support stable rankings. Common word extraction remains at 0.7% across SemPointer, BM25-4, the earlier implementation and full context under the 32 token generation cap. Its target lists ten words and may be truncated, so output length must be audited before drawing method conclusions.")
    doc.add_heading("5 Discussion and Limitations",1)
    para(doc,"SemPointer's supported contribution is an addressable, logged context budget interface. On the evaluated model and subsets, conditional block resolution reduces average active tokens and same run answer latency against fixed BM25-4, with no clearly established mean score difference. Stable addresses can help a serving system inspect what evidence was exposed for a particular answer. The experiment does not show that readable pointers are the cause of the efficiency gain; adaptive lexical selection may explain it.")
    para(doc,"The current suite lacks a pointer off control under the identical router, a strict per example iso token BM25 control, a held out seed evaluation and a semantic rescue ablation. Router thresholds were not selected on an independent development split. Selected LongBench tasks use a shared generation cap, and LoCoMo uses an adapted scorer. Results come from one quantized model on one GPU. The tested text resolution path does not validate a cached key value substrate. Each of these limits should narrow the interpretation of the numbers, especially when considering deployment outside the measured query mix.")
    para(doc,"The most informative next experiment is to hold the selected IDs and payload budget fixed while removing unresolved pointer entries. A held out seed should then test frozen router thresholds, and longer output caps should revisit common word extraction and affected LongBench tasks. Repeated query traces can charge registry construction once across multiple answers. These controls would identify whether the address layer adds value beyond adaptive passage selection and whether the timing advantage survives realistic use.")
    doc.add_heading("5.1 Practical interpretation",2)
    para(doc,"The measured architecture suits a setting in which an operator needs a bounded prompt and a record of which stored blocks were exposed to the model. It gives that operator one simple decision per question: retain the two strongest lexical blocks, or spend more text budget when coverage or the ranking boundary raises concern. On the tested RULER set, the cheapest fixed two block system remains faster and shorter, and full context remains more accurate. Choosing SemPointer is therefore a budget decision based on the application's value for quality, tokens, latency and inspectability.")
    para(doc,"A production evaluation should replay real query sequences over persistent histories. It should report ingestion once, prompt construction, time to first token and complete response latency separately. It should also test whether the router's 0.80, 0.95 and 0.85 thresholds remain useful across document lengths, paraphrased questions and multi hop tasks. Until those controls are complete, the strong claim is confined to the measured quality and cost frontier shown here.")
    doc.add_heading("6 Conclusion",1)
    para(doc,"SemPointer couples stable block addresses with a two or expanded lexical route. In completed runs it substantially improves over the earlier attention fused implementation and uses fewer active tokens and less answer time than fixed four block BM25 at similar observed scores on three selected collections. Its value is an inspectable budget choice under modest hardware. Pointer specific causal value, accuracy equivalence, official benchmark standing and robustness across models remain open.")
    doc.add_heading("Appendix Architecture Workflow Specification",1)
    para(doc,"The workflow below fixes the operational order used by the measured system. It is included to make the address layer and routing policy unambiguous. It does not add a new experiment or claim about accuracy.")
    for step in [
        "Ingest history. Tokenize with the generator tokenizer, split into blocks of at most 512 tokens, assign stable IDs, write exact payload text, and generate one readable pointer of at most 16 tokens per block.",
        "Rank for a question. Score all registered blocks with BM25 and order IDs by score, breaking equal scores by the original block order. Protect the first two IDs.",
        "Measure support. Compute weighted coverage from distinct question terms present in protected blocks, using nonnegative inverse document frequency weights. Define coverage as zero if its denominator is zero.",
        "Choose candidates. Request up to four blocks if coverage is below 0.80, or if coverage is below 0.95 while the third to second BM25 score ratio is at least 0.85. Otherwise retain the protected two.",
        "Enforce the resolved text budget. Add requested candidates in rank order until adding another block would exceed 2,560 resolved payload tokens. Preserve protected candidates according to the implementation's selection order.",
        "Assemble and answer. Insert exact text for selected IDs, leave compact entries for unresolved IDs, add the question and instructions, and call the same generator used by all same run controls. Record selection route, IDs, tokens, time to first token and complete answer latency."
    ]:
        p=doc.add_paragraph(style="Normal");p.style=doc.styles["Normal"];p.paragraph_format.left_indent=Inches(.25);p.paragraph_format.first_line_indent=Inches(-.2)
        p.add_run("• "+step)
    para(doc,"A valid trace should preserve three invariants: every displayed ID resolves to the same stored payload, every selected payload appears exactly once in the mixed prompt, and logged active tokens come from the actual model input rather than the approximate budget equation. These checks distinguish a routing error from an answer generation failure and make later pointer off and iso token controls reproducible.")
    doc.add_heading("Appendix Control Definitions",1)
    para(doc,"A pointer off control should keep the BM25 ranking, route decision, resolved IDs, text cap, model and generation settings fixed for each example. It should then remove only the unresolved pointer entries from the prompt. Reporting both raw scores and paired differences would answer whether those entries add information beyond selected payload text. A separate iso token control should give the fixed BM25 comparator the same per example active token allowance, including framing overhead, and use a prespecified truncation rule. Keeping these controls separate matters because removing pointers changes the prompt budget, while matching prompt budgets can change which payloads fit.")
    para(doc,"For serving, the complete cost record should include one time history segmentation and registry construction, then per question ranking, routing, prompt assembly, time to first token and final response time. Repeating questions over each stored history would make the setup cost amortization measurable. The current latency result deliberately excludes ingestion and should remain labeled that way in any derivative plot or comparison.")
    doc.add_heading("References",1)
    refs=["Bai et al. (2024). LongBench: A bilingual, multitask benchmark for long context understanding. ACL, 3119–3137.","Dettmers et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. NeurIPS 36.","Hsieh et al. (2024). RULER: What's the real context size of your long context language models? COLM.","Lewis et al. (2020). Retrieval augmented generation for knowledge intensive NLP tasks. NeurIPS 33.","Maharana et al. (2024). Evaluating very long term conversational memory of LLM agents. ACL, 13851–13870.","Mohtashami and Jaggi (2023). Landmark attention: Random access infinite context length for transformers. NeurIPS 36.","Robertson and Zaragoza (2009). The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval 3(4), 333–389.","Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115."]
    for ref in refs:
        p=para(doc,ref)
        p.paragraph_format.space_after=Pt(2)
        p.paragraph_format.line_spacing=1.05
        for run in p.runs: run.font.size=Pt(9.5)
    core=doc.core_properties;core.title="SemPointer Adaptive Addressable Context for Long Context Question Answering";core.author="Viksit Garg";core.subject="Research manuscript";core.keywords="SemPointer adaptive retrieval long context"
    doc.save(OUT)
    print(OUT)


if __name__=="__main__":build()
