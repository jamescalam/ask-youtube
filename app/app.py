import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer

PINECONE_KEY = st.secrets["PINECONE_KEY"]  # app.pinecone.io

@st.experimental_singleton
def init_pinecone():
    pinecone.init(api_key=PINECONE_KEY, environment="us-west1-gcp")
    return pinecone.Index('youtube-search')
    
@st.experimental_singleton
def init_retriever():
    return SentenceTransformer("multi-qa-mpnet-base-dot-v1")

index = init_pinecone()
retriever = init_retriever()

def card(thumbnail: str, title: str, urls: list, contexts: list, starts: list, ends: list):
    meta = [(e, s, u, c) for e, s, u, c in zip(ends, starts, urls, contexts)]
    meta.sort(reverse=False)
    text_content = []
    current_start = 0
    current_end = 0
    for end, start, url, context in meta:
        # reformat seconds to timestamp
        time = start / 60
        mins = f"0{int(time)}"[-2:]
        secs = f"0{int(round((time - int(mins))*60, 0))}"[-2:]
        timestamp = f"{mins}:{secs}"
        if start < current_end and start > current_start:
            # this means it is a continuation of the previous sentence
            text_content[-1][0] = text_content[-1][0].split(context[:10])[0]
            text_content.append([f"[{timestamp}] {context.capitalize()}", url])
        else:
            text_content.append(["xxLINEBREAKxx", ""])
            text_content.append([f"[{timestamp}] {context}", url])
        current_start = start
        current_end = end
    html_text = ""
    for text, url in text_content:
        if text == "xxLINEBREAKxx":
            html_text += "<br>"
        else:
            html_text += f"<small><a href={url}>{text.strip()}... </a></small>"
    print(html_text)
    html = f"""
    <div class="container-fluid">
        <div class="row align-items-start">
            <div class="col-md-4 col-sm-4">
                <div class="position-relative">
                    <a href={urls[0]}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                </div>
            </div>
            <div  class="col-md-8 col-sm-8">
                <h2>{title}</h2>
            </div>
        <div>
            {html_text}
    <br><br>
    """
    return st.markdown(html, unsafe_allow_html=True)

    
st.write("""
# YouTube Q&A
Ask me a question!
""")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)

query = st.text_input("Search!", "")

if query != "":
    xq = retriever.encode([query]).tolist()
    xc = index.query(xq, top_k=5, include_metadata=True)
    
    results = {}
    order = []
    for context in xc['matches']:
        video_id = context['metadata']['url'].split('/')[-1]
        if video_id not in results:
            results[video_id] = {
                'title': context['metadata']['title'],
                'urls': [f"{context['metadata']['url']}?t={int(context['metadata']['start'])}"],
                'contexts': [context['metadata']['text']],
                'starts': [int(context['metadata']['start'])],
                'ends': [int(context['metadata']['end'])]
            }
            order.append(video_id)
        else:
            results[video_id]['urls'].append(
                f"{context['metadata']['url']}?t={int(context['metadata']['start'])}"
            )
            results[video_id]['contexts'].append(
                context['metadata']['text']
            )
            results[video_id]['starts'].append(int(context['metadata']['start']))
            results[video_id]['ends'].append(int(context['metadata']['end']))
    # now display cards
    for video_id in order:
        card(
            thumbnail=f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            title=results[video_id]['title'],
            urls=results[video_id]['urls'],
            contexts=results[video_id]['contexts'],
            starts=results[video_id]['starts'],
            ends=results[video_id]['ends']
        )