"""
Quiz Page - Take Quiz, View Results, and City Details in one page with pill navigation.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.st_base import initialize_session_state
from components.quiz_questions import (
    QUIZ_QUESTIONS,
    get_all_questions,
    get_categories,
    get_total_question_count,
)
from components.scoring import calculate_city_scores, get_city_details
from components.navigation import render_navigation
from utilities.results_store import save_result, load_results, delete_result

initialize_session_state("Find Your Spot - Quiz", show_header_title=False)
render_navigation()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_completed" not in st.session_state:
    st.session_state.quiz_completed = False
if "quiz_section" not in st.session_state:
    st.session_state.quiz_section = "Take Quiz"

SECTIONS = ["Take Quiz", "My Results", "City Details"]


def _switch_section(name: str):
    """Set the active pill section and rerun."""
    st.session_state.quiz_section = name
    st.session_state.pop("quiz_pill", None)
    st.rerun()


# ---------------------------------------------------------------------------
# Pill navigation
# ---------------------------------------------------------------------------
active = st.pills(
    "Quiz Navigation",
    SECTIONS,
    default=st.session_state.quiz_section,
    key="quiz_pill",
    label_visibility="collapsed",
)

# Sync session state when user clicks a pill directly
if active and active != st.session_state.quiz_section:
    st.session_state.quiz_section = active
    st.rerun()


# ===================================================================
# SECTION: Take Quiz
# ===================================================================
def _render_take_quiz():
    def count_answered():
        count = 0
        for q in get_all_questions():
            q_id = q["id"]
            q_type = q.get("type")
            if q_type == "sliders":
                if q_id in st.session_state.quiz_answers:
                    count += 1
            elif q_type == "slider":
                if q_id in st.session_state.quiz_answers:
                    count += 1
            else:
                if q_id in st.session_state.quiz_answers and st.session_state.quiz_answers[q_id]:
                    count += 1
        return count

    def render_slider_question(question):
        q_id = question["id"]
        q_type = question.get("type")
        st.markdown(f"**{question['question']}**")

        if q_type == "sliders":
            sliders = question.get("sliders", [])
            if q_id not in st.session_state.quiz_answers:
                st.session_state.quiz_answers[q_id] = {}
            cols = st.columns(len(sliders))
            for i, slider in enumerate(sliders):
                with cols[i]:
                    slider_id = slider["id"]
                    current_val = st.session_state.quiz_answers[q_id].get(
                        slider_id, slider.get("default", 5)
                    )
                    slider_kwargs = {
                        "label": slider["label"],
                        "min_value": slider.get("min", 0),
                        "max_value": slider.get("max", 10),
                        "value": current_val,
                        "key": f"slider_{q_id}_{slider_id}",
                    }
                    if "step" in slider:
                        slider_kwargs["step"] = slider["step"]
                    if "format" in slider:
                        slider_kwargs["format"] = slider["format"]
                    new_val = st.slider(**slider_kwargs)
                    st.session_state.quiz_answers[q_id][slider_id] = new_val
        else:
            default = question.get("default", question.get("min", 0))
            current_val = st.session_state.quiz_answers.get(q_id, default)
            format_str = question.get("format", "%d")
            new_val = st.slider(
                "",
                min_value=question.get("min", 0),
                max_value=question.get("max", 100),
                value=current_val,
                step=question.get("step", 1),
                format=format_str,
                key=f"slider_{q_id}",
                label_visibility="collapsed",
            )
            st.session_state.quiz_answers[q_id] = new_val
        st.divider()

    def render_question(question, category_key):
        q_id = question["id"]
        q_type = question.get("type")

        if q_type in ["slider", "sliders"]:
            render_slider_question(question)
            return

        if q_id not in st.session_state.quiz_answers:
            st.session_state.quiz_answers[q_id] = [] if question.get("multi_select") else None

        st.markdown(f"**{question['question']}**")

        if question.get("multi_select"):
            raw = st.session_state.quiz_answers[q_id]
            # Migrate stale single-select string values to list
            if isinstance(raw, str):
                raw = [raw] if raw else []
            current_selections = raw or []
            with st.container(horizontal=True):
                for value, label in question["options"]:
                    checked = value in current_selections
                    if st.checkbox(label, value=checked, key=f"{q_id}_{value}"):
                        if value not in current_selections:
                            current_selections.append(value)
                    else:
                        if value in current_selections:
                            current_selections.remove(value)
            st.session_state.quiz_answers[q_id] = current_selections
        else:
            options = [opt[1] for opt in question["options"]]
            option_values = [opt[0] for opt in question["options"]]
            current_value = st.session_state.quiz_answers[q_id]
            current_index = option_values.index(current_value) if current_value in option_values else None

            selected = st.radio(
                "Select one:",
                options,
                index=current_index,
                key=f"radio_{q_id}",
                horizontal=True,
                label_visibility="collapsed",
            )
            if selected:
                selected_idx = options.index(selected)
                st.session_state.quiz_answers[q_id] = option_values[selected_idx]
        st.divider()

    # -- Page content --
    total_questions = get_total_question_count()
    answered = count_answered()
    progress = answered / total_questions if total_questions > 0 else 0

    with st.sidebar:
        st.markdown("Answer these questions to discover cities that match your lifestyle preferences.")
        st.progress(progress, text=f"Progress: {answered}/{total_questions} questions answered")
        if answered == total_questions:
            if st.button("See Results", type="primary", use_container_width=True):
                st.session_state.quiz_completed = True
                _switch_section("My Results")
        if st.button("Reset All Answers", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_completed = False
            st.rerun()

    st.title("Find Your Perfect City")

    for category in get_categories():
        questions = QUIZ_QUESTIONS[category]
        category_answered = 0
        for q in questions:
            q_id = q["id"]
            q_type = q.get("type")
            if q_type in ["slider", "sliders"]:
                if q_id in st.session_state.quiz_answers:
                    category_answered += 1
            elif q_id in st.session_state.quiz_answers and st.session_state.quiz_answers[q_id]:
                category_answered += 1

        with st.expander(
            f"{category} ({category_answered}/{len(questions)})",
            expanded=True,
        ):
            for question in questions:
                render_question(question, category)

    # Bottom navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if answered == total_questions:
            st.success("All questions answered!")
            if st.button("View My Results", type="primary", use_container_width=True, key="bottom_results"):
                st.session_state.quiz_completed = True
                _switch_section("My Results")
        elif answered > total_questions * 0.5:
            st.info(f"Good progress! {total_questions - answered} questions remaining.")
            if st.button("View Preliminary Results", use_container_width=True):
                _switch_section("My Results")
        elif answered > 0:
            st.info(f"Keep going! {total_questions - answered} questions remaining.")
        else:
            st.info("Start answering questions above to discover your ideal cities.")


# ===================================================================
# SECTION: My Results
# ===================================================================
def _render_results():
    # Check if quiz has been started
    if "quiz_answers" not in st.session_state or not st.session_state.quiz_answers:
        st.title("No Quiz Results Yet")
        st.warning("Please complete the quiz first to see your city recommendations.")
        if st.button("Take the Quiz", type="primary"):
            _switch_section("Take Quiz")
        return

    answered = sum(1 for v in st.session_state.quiz_answers.values() if v)
    total = get_total_question_count()

    if answered < total * 0.5:
        st.title("Quiz Incomplete")
        st.warning(f"Please answer more questions to get accurate results. ({answered}/{total} answered)")
        if st.button("Continue Quiz", type="primary"):
            _switch_section("Take Quiz")
        return

    st.title("Your Top City Matches")
    st.markdown(f"Based on your {answered} answers, here are the cities that best match your preferences.")

    # Calculate scores
    with st.spinner("Finding your perfect cities..."):
        top_cities = calculate_city_scores(st.session_state.quiz_answers, top_n=10)

    st.session_state.top_cities = top_cities

    # Save / history controls
    with st.container(horizontal=True):
        with st.popover("Save Results"):
            save_title = st.text_input("Title", placeholder="e.g. Warm climate, low cost")
            if st.button("Save", type="primary", key="confirm_save"):
                if not save_title.strip():
                    st.warning("Please enter a title.")
                else:
                    result_id = save_result(
                        st.session_state.quiz_answers,
                        top_cities,
                        questions_version=total,
                        title=save_title.strip(),
                    )
                    st.success(f"Saved: {save_title.strip()}")

        saved = load_results()
        if saved:
            with st.popover("Load Saved Results"):
                for r in saved:
                    title = r.get("title") or r["id"]
                    ts = r["timestamp"][:16].replace("T", " ")
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        if st.button(f"{title}  ({ts})", key=f"load_{r['id']}"):
                            st.session_state.quiz_answers = r["quiz_answers"]
                            st.session_state.top_cities = r["top_cities"]
                            st.rerun()
                    with col_b:
                        if st.button("X", key=f"del_{r['id']}"):
                            delete_result(r["id"])
                            st.rerun()

    # Map
    df_map = pd.DataFrame(top_cities)
    df_map["rank"] = range(1, len(df_map) + 1)
    df_map["marker_size"] = 30 - (df_map["rank"] - 1) * 2

    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        size="marker_size",
        color="total_score",
        color_continuous_scale="Greens",
        range_color=[df_map["total_score"].min() - 5, df_map["total_score"].max()],
        hover_name="name",
        hover_data={
            "state": True,
            "total_score": ":.1f",
            "rank": True,
            "population": ":,",
            "marker_size": False,
            "lat": False,
            "lon": False,
        },
        labels={
            "total_score": "Match Score",
            "state": "State",
            "rank": "Rank",
            "population": "Population",
        },
        zoom=3,
        center={"lat": 39.8283, "lon": -98.5795},
        mapbox_style="carto-darkmatter",
    )
    fig.update_layout(
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar={"title": "Match<br>Score", "ticksuffix": "%"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": True})

    st.divider()

    # City cards
    st.subheader("Your Top 10 Cities")

    for i, city in enumerate(top_cities):
        rank = i + 1
        col1, col2, col3 = st.columns([1, 4, 2])

        with col1:
            if rank <= 3:
                medals = {1: "\U0001f947", 2: "\U0001f948", 3: "\U0001f949"}
                st.markdown(f"### {medals[rank]}")
            else:
                st.markdown(f"### #{rank}")

        with col2:
            st.markdown(f"### {city['name']}, {city['state']}")
            scores = city["scores"]
            score_cols = st.columns(4)
            with score_cols[0]:
                st.metric("Climate", f"{scores['climate']:.0f}")
            with score_cols[1]:
                st.metric("Cost", f"{scores['cost_taxes']:.0f}")
            with score_cols[2]:
                st.metric("Outdoor", f"{scores['outdoor']:.0f}")
            with score_cols[3]:
                st.metric("Lifestyle", f"{scores['lifestyle']:.0f}")

            stats = city["stats"]
            caption_parts = [
                f"Population: {city['population']:,}",
                f"Weather: {stats['avg_temp_summer']:.0f}F/{stats['avg_temp_winter']:.0f}F",
                f"Housing: {stats['cost_of_living_index']:.0f}",
                f"Home Price: ${stats['median_home_price']:,}",
            ]
            if "estimated_taxes" in city and pd.notna(city["estimated_taxes"].get("total")):
                caption_parts.append(f"Est. Taxes: ${city['estimated_taxes']['total']:,.0f}/yr")
            st.caption(" | ".join(caption_parts))

        with col3:
            st.markdown(f"### {city['total_score']:.0f}%")
            st.caption("Match Score")
            if st.button("View Details", key=f"details_{city['city_id']}", use_container_width=True):
                st.session_state.selected_city = city["city_id"]
                _switch_section("City Details")

        st.divider()

    # Bottom nav
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Retake Quiz"):
            _switch_section("Take Quiz")
    with col3:
        if st.button("Modify Answers"):
            _switch_section("Take Quiz")


# ===================================================================
# SECTION: City Details
# ===================================================================
def _render_city_details():
    if "selected_city" not in st.session_state:
        st.title("No City Selected")
        st.warning("Please select a city from your results to view details.")
        if st.button("View Results", type="primary"):
            _switch_section("My Results")
        return

    city_id = st.session_state.selected_city
    city = get_city_details(city_id)

    if city is None:
        st.error("City not found. Please try again.")
        if st.button("Back to Results"):
            _switch_section("My Results")
        return

    # Find score data
    city_score_data = None
    if "top_cities" in st.session_state:
        for c in st.session_state.top_cities:
            if c["city_id"] == city_id:
                city_score_data = c
                break

    if city_score_data is None and "quiz_answers" in st.session_state:
        all_scores = calculate_city_scores(st.session_state.quiz_answers, top_n=100)
        for c in all_scores:
            if c["city_id"] == city_id:
                city_score_data = c
                break

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"{city['name']}, {city['state']}")
        metro = city.get("cbsa_name", "")
        metro_pop = city.get("metro_pop")
        subtitle = f"**{city['region']} Region** | Population: {city['population']:,}"
        if metro and pd.notna(metro_pop) and metro_pop > 0:
            subtitle += f" | Metro: {metro} ({int(metro_pop):,})"
        st.markdown(subtitle)
    with col2:
        if city_score_data:
            st.metric("Match Score", f"{city_score_data['total_score']:.0f}%")

    st.divider()

    # Radar chart
    if city_score_data:
        st.subheader("Score Breakdown")
        scores = city_score_data["scores"]
        categories = list(scores.keys())
        values = list(scores.values())

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=[c.replace("_", " ").title() for c in categories] + [categories[0].replace("_", " ").title()],
            fill="toself",
            fillcolor="rgba(0, 176, 80, 0.3)",
            line=dict(color="rgb(0, 176, 80)", width=2),
            name="Score",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # City profile
    st.subheader("City Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Climate")
        st.metric("Summer Temp", f"{city['avg_temp_summer']:.0f} F")
        st.metric("Winter Temp", f"{city['avg_temp_winter']:.0f} F")
        st.metric("Sunny Days/Year", f"{city['sunny_days']}")
        st.metric("Annual Rainfall", f"{city['annual_rainfall']:.0f} in")
        st.metric("Annual Snowfall", f"{city['annual_snow']:.0f} in")

    with col2:
        st.markdown("#### Cost of Living")
        st.metric("Housing Cost Index", f"{city['cost_of_living_index']:.0f}", help="Based on home prices + rent (100 = national avg)")
        goods_rpp = city.get("goods_rpp")
        if pd.notna(goods_rpp):
            st.metric("Goods Price Index", f"{goods_rpp:.1f}", help="BEA RPP — cost of consumer goods (100 = national avg)")
        st.metric("Median Home Price", f"${city['median_home_price']:,}")
        st.metric("State Income Tax", "None" if city["no_income_tax_state"] else f"{city['state_income_tax_rate']:.1f}%")
        st.metric("Property Tax Rate", f"{city['avg_property_tax_rate']:.2f}%")
        st.metric("Sales Tax", f"{city['state_sales_tax_rate']:.1f}%")

        # Show personalized tax estimate if user provided financials
        if city_score_data and "estimated_taxes" in city_score_data:
            taxes = city_score_data["estimated_taxes"]
            if pd.notna(taxes.get("total")):
                st.markdown("#### Your Estimated Taxes")
                st.metric("State Income Tax", f"${taxes['income_tax']:,.0f}/yr")
                st.metric("Property Tax", f"${taxes['property_tax']:,.0f}/yr")
                st.metric("Sales Tax", f"${taxes['sales_tax']:,.0f}/yr")
                st.metric("Total Estimated Taxes", f"${taxes['total']:,.0f}/yr")

    with col3:
        st.markdown("#### Livability")
        st.metric("Walkability", f"{city['walkability_score']}/100")
        st.metric("Transit Score", f"{city['transit_score']}/100")
        st.metric("Crime Rate", f"{city['crime_rate_per_1000']:.1f}/1000")
        st.metric("Commute Time", f"{city['mean_commute_minutes']:.0f} min" if pd.notna(city.get('mean_commute_minutes')) else "N/A")
        st.metric("Unemployment", f"{city['unemployment_rate']:.1f}%")

    st.divider()

    # Geography & Recreation
    st.subheader("Geography & Recreation")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Landscape Features")
        features = []
        if city["has_mountains"]:
            features.append("Mountains nearby")
        if city["has_ocean"]:
            features.append("Ocean/beach access")
        lake_mi = city.get("nearest_boatable_lake_miles", 999)
        if lake_mi < 30:
            features.append(f"Boatable lake {lake_mi:.0f} mi away")
        if city["has_desert"]:
            features.append("Desert landscape")
        for feature in features:
            st.markdown(f"- {feature}")
        if not features:
            st.markdown("- Plains/prairie landscape")

    with col2:
        st.markdown("#### Outdoor Recreation")
        st.markdown(f"- Ski resort: {city['ski_resort_distance_miles']:.0f} miles")
        st.markdown(f"- Hiking trails: {city['hiking_trails_count']}")
        st.markdown(f"- Mountain biking: {city['mountain_biking_trails']} trails")
        st.markdown(f"- Rock climbing areas: {city['rock_climbing_areas_nearby']}")
        st.markdown(f"- National parks (100mi): {city['national_parks_within_100mi']}")
        st.markdown(f"- State parks nearby: {city['state_parks_nearby']}")

    st.divider()

    # Culture & Entertainment
    st.subheader("Culture & Entertainment")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Arts & Entertainment")
        st.markdown(f"- Museums: {city['museums_count']}")
        st.markdown(f"- Performing arts venues: {city['performing_arts_venues']}")
        st.markdown(f"- Concert venues: {city['concert_venue_count']}")

    with col2:
        st.markdown("#### Sports & Education")
        major_teams = int(city.get('major_pro_teams', 0))
        minor_teams = int(city.get('minor_pro_teams', 0))
        leagues = city.get('pro_leagues', '')
        st.markdown(f"- Major pro teams: {major_teams}" + (f" ({leagues})" if leagues else ""))
        st.markdown(f"- Minor league teams: {minor_teams}")
        st.markdown(f"- Universities: {int(city['university_count'])}")
        st.markdown(f"- Major university: {'Yes' if city['has_major_university'] else 'No'}")
        st.markdown(f"- Community college: {'Yes' if city['has_community_college'] else 'No'}")
        if city.get('college_town_score', 0) >= 0.05:
            st.markdown(f"- College town score: {city['college_town_score']:.0%}")

    st.divider()

    # Transportation & Economy
    st.subheader("Transportation & Economy")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Transportation")
        st.markdown(f"- Airport distance: {city['airport_distance_miles']:.0f} miles")
        st.markdown(f"- Airline hub: {'Yes' if city['is_airline_hub'] else 'No'}")
        st.markdown(f"- Direct flight destinations: {city['direct_flight_destinations_count']}")

    with col2:
        st.markdown("#### Economy")
        st.markdown(f"- Median income: ${city['median_household_income']:,}")
        st.markdown(f"- Median rent: ${city['median_gross_rent']:,}")
        st.markdown(f"- Poverty rate: {city['poverty_rate']:.1f}%")
        pct_wfh = city.get('pct_work_from_home')
        if pd.notna(pct_wfh):
            st.markdown(f"- Work from home: {pct_wfh:.1f}%")

    st.divider()

    # External links
    st.subheader("Learn More")
    col1, col2, col3 = st.columns(3)
    city_name_encoded = city["name"].replace(" ", "+")
    state_encoded = city["state"]

    with col1:
        st.link_button(
            "Search on Zillow",
            f"https://www.zillow.com/homes/{city_name_encoded},-{state_encoded}",
            use_container_width=True,
        )
    with col2:
        st.link_button(
            "Google Maps",
            f"https://www.google.com/maps/place/{city_name_encoded},+{state_encoded}",
            use_container_width=True,
        )
    with col3:
        st.link_button(
            "Wikipedia",
            f"https://en.wikipedia.org/wiki/{city['name'].replace(' ', '_')},_{state_encoded}",
            use_container_width=True,
        )

    # Back to results
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Back to Results"):
            _switch_section("My Results")
    with col3:
        if st.button("Retake Quiz"):
            _switch_section("Take Quiz")


# ===================================================================
# Render the active section
# ===================================================================
section = st.session_state.quiz_section

if section == "Take Quiz":
    _render_take_quiz()
elif section == "My Results":
    _render_results()
elif section == "City Details":
    _render_city_details()
