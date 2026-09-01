from __future__ import annotations

import streamlit as st

from src.rag import config, usage
from src.rag.pipeline import answer as run_pipeline
from src.rag.ui_helpers import (
    ROTATING_EXAMPLES_HTML, SPACE_BACKGROUND_HTML, THEME_CSS, anirdesh_url, build_quote_preview,
    build_source_label, format_answer_html, load_anirdesh_vachno_map,
)

USAGE_CAVEAT = (
    'Estimated from local usage tracking only -- resets if this app restarts '
    '(redeploy, waking from sleep) and may undercount if others are using '
    'this same deployment. Not an authoritative count from Google.'
)

st.set_page_config(page_title='Ask Vachanamrut', page_icon='✨', layout='centered')
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown(SPACE_BACKGROUND_HTML, unsafe_allow_html=True)

st.markdown('<div class="app-title">Ask Vachanamrut</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Model')
    model = st.selectbox('Model', options=config.AVAILABLE_MODELS, label_visibility='collapsed')
    remaining = usage.requests_remaining_today(model)
    if remaining is not None:
        limit = config.MODEL_DAILY_LIMITS[model]
        st.metric('Requests left today', f'{remaining} / {limit}', help=USAGE_CAVEAT)
    else:
        st.metric('Requests today', usage.requests_used_today(model), help=USAGE_CAVEAT)

    st.divider()

    with st.expander('Advanced options'):
        k = st.slider('Sources to retrieve (k)', min_value=1, max_value=10, value=config.K_FINAL)
        neighbors = st.slider(
            'Neighbor chunks per source', min_value=0, max_value=3, value=config.EXPAND_NEIGHBORS,
        )
        max_expanded_tokens = st.slider(
            'Max expanded tokens per source', min_value=200, max_value=3000,
            value=config.MAX_EXPANDED_TOKENS, step=100,
        )
        temperature = st.slider(
            'Temperature', min_value=0.0, max_value=1.0, value=config.GENERATION_TEMPERATURE, step=0.05,
        )

if 'result' not in st.session_state:
    st.session_state.result = None

st.markdown(ROTATING_EXAMPLES_HTML, unsafe_allow_html=True)

with st.form('query_form'):
    query = st.text_input(
        'Ask a question about the Vachanamrut', placeholder='', label_visibility='collapsed',
    )
    submitted = st.form_submit_button('Ask')

if submitted and query.strip():
    with st.spinner('Consulting the discourses...'):
        try:
            st.session_state.result = run_pipeline(
                query, k=k, neighbors=neighbors, max_expanded_tokens=max_expanded_tokens,
                temperature=temperature, model=model,
            )
            usage.record_request(model)
        except Exception as exc:
            st.session_state.result = None
            st.error(f"Couldn't generate an answer: {exc}")

result = st.session_state.result
if result:
    with st.container(key='answer_box'):
        st.markdown(
            format_answer_html(result['answer'], result['cited'], result['sources']),
            unsafe_allow_html=True,
        )

    displayed_sources = result['sources'][:config.MAX_DISPLAYED_PASSAGES]
    with st.expander(f'Show retrieved passages ({len(displayed_sources)})'):
        vachno_map = load_anirdesh_vachno_map()
        for source in displayed_sources:
            st.markdown(f'**{build_source_label(source)}**')
            if source['chunk_id'] in result['cited']:
                st.caption('Cited in the answer above')
            st.markdown(build_quote_preview(source['text']))
            vachno = vachno_map.get(source['parent_id'])
            if vachno is not None:
                st.markdown(f'[Read the full discourse on Anirdesh]({anirdesh_url(vachno)})')
            st.divider()
