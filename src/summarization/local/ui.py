"""Presentation only; model transport is injected by the application."""
CSS = """
:root { --accent: #59d3bf; }
.gradio-container { max-width: 1320px !important; margin: auto !important; padding: 30px 28px 24px !important; }
#masthead { border-bottom: 1px solid #293342; padding-bottom: 24px; margin-bottom: 8px; }
.brand { color: #83decd; font-size: 11px; font-weight: 700; letter-spacing: .22em; margin-bottom: 14px; }
.hero-title { font-size: clamp(30px,4vw,46px); line-height: 1.1; font-weight: 650; letter-spacing: -.04em; color: #f3f6fa; margin: 0 0 12px; }
.hero-subtitle { color: #a7b4c5; font-size: 15px; max-width: 700px; line-height: 1.65; }
.live-pill { display: inline-block; background: #153e36; color: #9be6d5; padding: 6px 11px; border-radius: 20px; font-size: 12px; margin-top: 16px; }
.workspace-panel { border: 1px solid #293342 !important; border-radius: 18px !important; padding: 22px !important; background: #111923 !important; }
.panel-eyebrow { color: #7f98a7; font-size: 11px; font-weight: 700; letter-spacing: .15em; }
.panel-title { font-size: 21px; font-weight: 600; color: #f3f6fa; margin: 7px 0 16px; }
#generate { background: #76dfcb !important; color: #0b2521 !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; min-height: 48px; }
#generate:hover { background: #a0eddf !important; }
#source textarea, #summary textarea { font-size: 15px !important; line-height: 1.8 !important; }
#summary textarea { font-size: 18px !important; }
.metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 12px 0; }
.metric { background: #131e2a; border: 1px solid #293342; padding: 16px 12px; border-radius: 12px; }
.metric strong { display: block; color: #d7fff4; font-size: 25px; line-height: 1.2; }
.metric span { display: block; color: #91a5b9; font-size: 11px; margin-top: 8px; }
.notice { color: #a8bbcc; font-size: 12px; padding-top: 5px; line-height: 1.6; }
.footer-note { border-top: 1px solid #293342; padding-top: 16px; margin-top: 18px; color: #8b9cb0; font-size: 12px; line-height: 1.7; }
#wordcount { color: #93a9bc; font-size: 12px; }
@media(max-width: 700px) { .gradio-container { padding: 18px 12px !important; } .workspace-panel { padding: 15px !important; } .metrics { grid-template-columns: repeat(2,1fr); } }
"""

EXAMPLES = [["The world's biggest carmaker said it had identified more than 400 reports of the problem, with "
  'the bulk of them occurring in Japan and North America.\n'
  'No accidents or injuries have been reported over the defect, Toyota said.\n'
  'The recall comes weeks after Toyota halted the sale of some car models in the US over '
  'problematic seat heaters.\n'
  'It also deals a blow to the Japanese carmaker, which has seen its reputation suffer following a '
  'series of mass recalls in recent years.\n'
  'The latest recall affects the current generation Prius sedans made since March 2009.\n'
  'The Prius was first released in 1997 and is one of the most popular hybrid vehicles on the '
  'market.\n'
  'The new software problem can set off warning lights, and will "probably cause the vehicle to '
  'enter a failsafe" mode, which will see the car\'s driving power reduced.\n'
  '"In limited cases, the hybrid system might shut down and the vehicle will stop, perhaps while '
  'being driven," Toyota said.\n'
  "In 2010, Toyota's share price was hit after more than 10 million of its vehicles were recalled "
  'over unintended acceleration issues.\n'
  'Despite its problems, Toyota is forecasting a record profit for this year because of the '
  'effects of the weaker Japanese currency, which makes its cars cheaper abroad.\n'
  'Last year, it was the top-selling global carmaker, with worldwide deliveries rising 2.4% to '
  'nearly 10 million vehicles.'],
 ['Glasgow reported 606,117 people travelling through the airport last month, a 14% increase on '
  'the same period in 2014.\n'
  'Edinburgh Airport reported a 6.8% rise, to 791,975 passengers.\n'
  "For Glasgow it was the airport's 25th consecutive month of growth and its busiest March in "
  'seven years. International traffic grew by 22.3%.\n'
  'This was mainly due to strong demand for European destinations, the airport said.\n'
  'Domestic traffic at Glasgow grew by 8.2% with British Airways, Flybe and Loganair all reporting '
  'a strong demand for London and regional services.\n'
  'Amanda McMillan, managing director of Glasgow Airport, said: "2015 has got off to a tremendous '
  'start and we are continuing to enjoy significant increases in our passenger numbers as a result '
  'of our popular and growing list of destinations."\n'
  'At Edinburgh Airport, domestic passenger numbers rose by 8.7% in March to 432,935.\n'
  'International traffic increased by 4.7%, with a total of 359,040 passengers travelling.\n'
  'Gordon Dewar, chief executive of Edinburgh Airport, said: "March was a strong month for us and '
  'we saw a significant increase in the number of passengers travelling through our airport."'],
 ['According to Gospel accounts, The Last Supper is the final meal that Jesus shared with his '
  'Apostles before his crucifixion.\n'
  'For centuries, it has been the subject of artistic interpretation, most famously by Leonardo da '
  'Vinci in his 15th Century mural.\n'
  "Now Iain Campbell, artist-in-residence at Glasgow's St George's Tron, Church of Scotland, has "
  'created a new Glaswegian version of the Last Supper featuring men who find themselves in '
  'difficult financial or personal circumstances.\n'
  'It was commissioned as part of a social enterprise project between the church and Glasgow City '
  'Mission to run a cafe in the church.\n'
  'The artist began his oil painting in September and has been working on it since in full public '
  'view.\n'
  'He says his previous work featured politicians Nicola Sturgeon, Ruth Davidson and Johann Lamont '
  'and he had wondered whether his Last Supper should feature well-known faces.\n'
  'However, he decided to use real people to highlight the work of the City Mission.\n'
  'Mr Campbell says: "Painting this in public, folks continually came up to me to chat about it.\n'
  '"Again and again they said they were surprised that they look like real people rather than some '
  'idealised version of the Last Supper.\n'
  '"There\'s a sense that there is some real raw stories behind the faces in the painting."\n'
  'Mr Campbell adds: "We decided to call the painting Our Last Supper. It was based on something '
  'one of the guys had said to me.\n'
  '"He said: \'I suppose for any one of us this might be our last supper.\'"\n'
  'John Wallace is one of the men featured in the painting.\n'
  'He was unemployed and finding life tough when the project started.\n'
  'He says: "A couple of bits of depression and things like that, problems getting the better of '
  'you.\n'
  '"Most of the gentlemen at this table have all come from the same sort of background. So you are '
  'not on your self."\n'
  'John says he values the support and friendship of the City Mission and life for him is looking '
  'up.\n'
  'He is picking up work in restaurant kitchens and has moved into his own place.\n'
  'Arthur Curtis, who is featured in the foreground of the painting, says: "Could it be you that '
  'it could happen to?\n'
  '"Could it be someone you know that it could happen to?\n'
  '"You\'ve just got to face up to it."\n'
  'Arthur says he hopes people will pop in for a coffee and enjoy a piece of art that focuses on '
  'those so often pushed to the margins of society.\n'
  'The Reverend Alastair Duncan, minister of the St George\'s Tron, says: "One of the questions '
  "Iain was often asked was 'Who is Jesus in the painting?'.\n"
  '"He would quote the parable of the sheep and the goats which Jesus told in Matthew\'s gospel in '
  'the context of which he said \'whatever you do for the least of these you do for me\'."\n'
  'The minister added: "The people who Glasgow City Mission supports may be in hostel '
  'accommodation, may have their own place or may be genuinely homeless but they are people who in '
  'some ways have been vulnerable or are finding a way out of that.\n'
  '"They just need that consistency of care and support for them whatever stage they are in."']]

def metric_cards(source=0, summary=0, elapsed=0):
    reduction = 100 * (1 - summary / source) if source else 0
    values = [(str(source), "ARTICLE WORDS"), (str(summary), "SUMMARY WORDS"),
              (f"{reduction:.0f}%", "SHORTER"), (f"{elapsed:.1f}s", "TOTAL TIME")]
    return '<div class="metrics">' + ''.join(
        f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in values) + '</div>'

def build_interface(gr, summarize):
    theme = gr.themes.Base(primary_hue="teal", neutral_hue="slate", font=["Segoe UI", "Arial", "sans-serif"]).set(
        body_background_fill="#0b1119", body_background_fill_dark="#0b1119",
        body_text_color="#e5edf5", body_text_color_dark="#e5edf5",
        block_background_fill="#111923", block_background_fill_dark="#111923",
        block_border_color="#293342", block_border_color_dark="#293342",
        input_background_fill="#0d151f", input_background_fill_dark="#0d151f",
        input_border_color="#314154", input_border_color_dark="#314154",
        input_placeholder_color="#8395a8", input_placeholder_color_dark="#8395a8")
    with gr.Blocks(title="NeGm | Text Summarization", theme=theme, css=CSS) as demo:
        gr.HTML('<div id="masthead"><div class="brand">NeGm</div>'
                '<h1 class="hero-title">Less text. More meaning.</h1>'
                '<div class="hero-subtitle">Turn English articles into concise summaries that capture the main event and its essential context.</div>'
                '<span class="live-pill">● Local model ready</span></div>')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=320, elem_classes="workspace-panel"):
                gr.HTML('<div class="panel-eyebrow">01 / SOURCE</div><div class="panel-title">Start with your article</div>')
                article = gr.Textbox(label="Article", show_label=False, placeholder="Paste an English article here…", lines=15, elem_id="source")
                count = gr.Markdown("0 words", elem_id="wordcount")
                review = gr.Checkbox(label="Review against source · experimental", value=False,
                                     info="Adds a second pass to check the draft against the article. Slower; not a guarantee of factual accuracy.")
                with gr.Row():
                    button = gr.Button("Generate summary", variant="primary", elem_id="generate", scale=3)
                    clear = gr.Button("Clear", scale=1)
            with gr.Column(scale=1, min_width=320, elem_classes="workspace-panel"):
                gr.HTML('<div class="panel-eyebrow">02 / SUMMARY</div><div class="panel-title">The essential story</div>')
                summary = gr.Textbox(label="Summary", show_label=False, placeholder="Your summary will appear here.",
                                     lines=8, interactive=False, show_copy_button=True, elem_id="summary")
                metrics = gr.HTML(metric_cards())
                status = gr.Markdown("Ready. Target: one or two sentences, at most 40 words.")
                download = gr.DownloadButton("Download summary", value=None, interactive=False)
                with gr.Accordion("Draft and runtime details", open=False):
                    draft = gr.Textbox(label="First draft", interactive=False, lines=4)
                    details = gr.JSON(label="Processing details")
        with gr.Accordion("Try a sample article", open=False):
            gr.Markdown("Fictional examples covering transport, research uncertainty and a sports comeback.")
            gr.Examples(examples=EXAMPLES, inputs=article, label="Sample articles", examples_per_page=3)
        gr.HTML('<div class="footer-note">Runs on your computer · Trained Qwen3-14B · English news summarization<br>'
                'Review names, numbers and important claims against the source before sharing.</div>')
        article.change(lambda value: f"{len(value.split()):,} words", inputs=article, outputs=count, queue=False)
        button.click(summarize, inputs=[article, review], outputs=[summary, metrics, status, draft, details, download],
                     concurrency_limit=1, concurrency_id="summary")
        clear.click(lambda: ("", "", metric_cards(), "Ready.", "", {}, gr.update(value=None, interactive=False)),
                    outputs=[article, summary, metrics, status, draft, details, download], queue=False)
    return demo
