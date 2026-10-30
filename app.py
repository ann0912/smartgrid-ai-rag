import streamlit as st
import pymongo
from google import genai
from google.genai import types

st.set_page_config(page_title='SmartGrid AI', page_icon='⚡')
st.title('⚡ SmartGrid AI')
st.caption('Asistente RAG basado en el PDF de IA aplicada a gestión energética, Smart City y Smart Grid')

@st.cache_resource
def get_clients():
    api_key = st.secrets['app']['GOOGLE_API_KEY']
    uri = st.secrets['app']['MONGODB_URI']
    return genai.Client(api_key=api_key), pymongo.MongoClient(uri, serverSelectionTimeoutMS=12000)

try:
    ai, mongo = get_clients()
    collection = mongo['pdf_embeddings_db']['smartgrid_vectors']
except Exception as exc:
    st.error('No se pudo inicializar la aplicación. Revisa los secretos y la conexión.')
    st.stop()

k = st.sidebar.select_slider('Fragmentos recuperados (k)', options=[3,5,10], value=5)
st.sidebar.caption('Compara k=3, k=5 y k=10 para la actividad.')

def retrieve(question, k):
    emb = ai.models.embed_content(model='gemini-embedding-001', contents=question,
        config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')).embeddings[0].values
    pipeline = [
        {'$vectorSearch': {'index':'vector_index','path':'embedding','queryVector':emb,
                           'numCandidates':100,'limit':k}},
        {'$project': {'_id':0,'id':1,'texto':1,'score':{'$meta':'vectorSearchScore'}}}
    ]
    return list(collection.aggregate(pipeline))

def answer(question, chunks):
    context = '\n\n'.join(f'[Fragmento {c.get("id", "?")}] {c["texto"]}' for c in chunks)
    prompt = f'''Eres SmartGrid AI, asistente especializado en el documento de clase sobre IA aplicada a gestión energética, Smart City y Smart Grid.
Responde en español usando EXCLUSIVAMENTE el CONTEXTO proporcionado. Si la respuesta no está sustentada en el contexto, indica: "No encuentro información suficiente en el documento para responder esa pregunta". No inventes datos ni fuentes. Sé claro y menciona los números de fragmentos que sustentan la respuesta cuando corresponda.

CONTEXTO:\n{context}\n\nPREGUNTA: {question}'''
    result = ai.models.generate_content(model='gemini-3.8-flash', contents=prompt)

    return result.text or 'No se generó una respuesta.'

if 'messages' not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg.get('chunks'):
            with st.expander('Fragmentos recuperados y scores'):
                for c in msg['chunks']:
                    st.write(f'Fragmento {c.get("id", "?")} | score: {c.get("score", 0):.5f}')
                    st.write(c['texto'])

if question := st.chat_input('Pregunta sobre Smart Grids y gestión energética...'):
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Buscando en el documento...'):
                chunks = retrieve(question, k)
                response = answer(question, chunks) if chunks else 'No encuentro información suficiente en el documento para responder esa pregunta.'
            st.markdown(response)
            with st.expander('Fragmentos recuperados y scores'):
                for c in chunks:
                    st.write(f'Fragmento {c.get("id", "?")} | score: {c.get("score", 0):.5f}')
                    st.write(c['texto'])
            st.session_state.messages.append({'role':'assistant','content':response,'chunks':chunks})

        except Exception:
            import logging
            logging.exception("Error durante la consulta RAG")
            st.error("No se pudo completar la consulta. Revisa los registros de la aplicación.")

