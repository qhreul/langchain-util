import streamlit as st
from streamlit_tags import st_tags
from PIL import Image

from rqle_ai_langchain_util import settings
from rqle_ai_langchain_util.utils import gui_util
from rqle_ai_langchain_util.llms import llm_guard_validator
from rqle_ai_langchain_util.apps.techtype_rocket.techtype_rocket import TechTypeRocket

import logging
from logging.config import dictConfig
dictConfig(settings.LOG_CONFIG)
logger = logging.getLogger(__name__)

APPLICATION_LOGO_PATH = 'img/techtype_rocket_logo.jpg'
APPLICATION_LOGO = Image.open(APPLICATION_LOGO_PATH)
APPLICATION_NAME = 'TechType Rocket'
APPLICATION_DESCRIPTION = f"""
    **{APPLICATION_NAME}**:copyright: is your one-stop platform designed to equip aspiring writers with the skills and 
    guidance to confidently create engaging tech blogs for diverse audiences.
"""

st.set_page_config(page_title=APPLICATION_NAME, page_icon=APPLICATION_LOGO,
                   layout='wide', initial_sidebar_state='collapsed')

# information about / help component of the application
st.sidebar = gui_util.set_page_sidebar(application_description=APPLICATION_DESCRIPTION)

# set the logo / title of the application
st.markdown(body=gui_util.set_page_header(application_name=APPLICATION_NAME,
                                           application_logo_path=APPLICATION_LOGO_PATH,
                                           background_color='#334246',
                                           text_color='#FFFFFF'),
            unsafe_allow_html=True)

# main body of the application
input, output, analytics = st.columns([1.5, 2, 0.5])
output.subheader('Generated Blog', divider='rainbow')
analytics.subheader('Analytics', divider='rainbow')

with ((input)):
    with st.form(key='blog_input'):
        st.subheader('Enter Blog Details', divider='rainbow')
        target_reading_time = st.number_input(label='Blog length (min)', min_value=5, step=1,
                                              help='Enter how long you want blog to be in terms of reading time')
        target_audience = st.selectbox(label='Target Audience', options=('Technology Leaders', 'Business Leaders', 'Novice'))
        # determine the focus of the blog
        topic = st.text_area(label='Topic', height=200)
        keywords = st_tags(label='Keywords', text='Press enter to add more keywords', maxtags=10)
        # determine whether the guard rails should be applied
        guard_rails_enabled = st.checkbox(label='Enable Guard Rails', value=True)
        # form submission button
        submit = st.form_submit_button(label='Generate', type='primary')

        # execute logic if submit button is clicked
        if submit:
            logger.debug(f'User input: target_reading_time={target_reading_time}, target_audience={target_audience}, topic={topic}, keywords={keywords}')
            # Checked that the required fields are not empty
            if (not topic and topic != '') or len(keywords) == 0:
                st.error('Please enter a few sentence describing the topic and at least one keyword to be written about.')
                logger.error('Error: No topics and / or keywords were entered by user.')
            else:
                with output:
                    with st.spinner('Generating blog...'):
                        try:
                            # check whether guard rails should be checked on the input
                            if guard_rails_enabled:
                                input_is_valid, input_error_msg = llm_guard_validator.validate_llm_input(input=topic)
                                # raise an exception if any of the guard rails are violated
                                if not input_is_valid:
                                    raise IOError(input_error_msg)
                            # execute the LLM to generate the blog
                            tech_type_rocket = TechTypeRocket(config_folder='techtype_rocket')
                            generated_blog_data = tech_type_rocket.invoke_chain(target_reading_time=target_reading_time,
                                                                           target_audience=target_audience,
                                                                           topic=topic,
                                                                           keywords=keywords)
                            # check whether guard rails should be checked on the output
                            if guard_rails_enabled:
                                output_is_valid, output_error_msg = llm_guard_validator.validate_llm_output(prompt=f'Generate a blog post about "{topic}" targeting {target_audience} '
                                                                                                                   f'with an estimated reading time of {target_reading_time} minutes.',
                                                                                                            output=generated_blog_data)
                                if not output_is_valid:
                                    raise IOError(output_error_msg)
                            markdown_tab, text_tab = st.tabs(['WYSIWYG', 'Markdown'])
                            with markdown_tab:
                                st.markdown(generated_blog_data)
                            with text_tab:
                                st.text(generated_blog_data)
                            # analytics component
                            with analytics:
                                st.subheader('Reading Time')
                                estimated_reading_time = llm_guard_validator.estimate_reading_time(generated_blog_data)
                                logger.debug(f'Estimated reading time of generated blog is {estimated_reading_time} minutes,'
                                             f'while the target was {target_reading_time} minutes.')
                                chart = gui_util.make_donut_graph(expected=target_reading_time, actual=estimated_reading_time)
                                st.plotly_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.error(f'An error occurred while generating the blog: {e}')
                            logger.error(f'Error occurred while generating the blog: {e}')

# information about the footer of the application (including logo and years)
st.markdown(body=gui_util.set_page_footer(), unsafe_allow_html=True)