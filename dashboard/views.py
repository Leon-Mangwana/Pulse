"""The four pages of the Pulse dashboard, in demo order: worklist, customer card, insights, ROI.

Text about a customer comes from the API's fixed templates (rule 8); this module only lays it out.
No raw feature name is rendered: tables go through dashboard/logic.py renames, cards use the
reason text and family label, drivers use family labels.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import api_client as api
from dashboard import logic

ACCENT = "#0B6E6E"
MUTED = "#9AA5B1"
PAGES: dict[str, object] = {}          # filled by app.py with the st.Page objects (for switch_page)

CSS = """
<style>
[data-testid="stMetricValue"] { font-size: 2.4rem; font-weight: 600; color: #0B6E6E; }
[data-testid="stMetricLabel"] { font-size: 0.95rem; }
h1 { font-size: 2.0rem !important; margin-bottom: 0.2rem; }
h2 { font-size: 1.35rem !important; margin-top: 0.6rem; }
h3 { font-size: 1.1rem !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 1rem; }
.pulse-chip { display:inline-block; padding: 0.1rem 0.6rem; border-radius: 999px;
              background: #E3F1F1; color: #0B6E6E; font-size: 0.85rem; margin-right: 0.4rem; }
.pulse-script { background: #F2F6F6; border-left: 4px solid #0B6E6E; padding: 0.7rem 1rem;
                border-radius: 4px; font-size: 1.02rem; }
</style>
"""

URGENCY_COLOURS = {"act this week": "#B42318", "act this month": "#B54708", "watch": MUTED}


def header(title: str, subtitle: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def open_card(ref: str) -> None:
    st.session_state["ref"] = ref
    if "card" in PAGES:
        st.switch_page(PAGES["card"])


def snapshot_caption(h: dict) -> str:
    return f"Snapshot {h['snapshot_date']}  |  {h['customers']:,} customers scored  |  all money in USD"


# ----------------------------------------------------------------------------- 1. worklist
def worklist() -> None:
    h = st.session_state["health"]
    header("Who to call this week", snapshot_caption(h))
    rows = api.worklist()

    f1, f2, f3 = st.columns([1, 1, 2])
    branch = f1.selectbox("Branch", ["All branches"] + sorted(rows["branch"].unique()))
    segment = f2.selectbox("Segment", ["All segments"] + sorted(rows["segment"].unique()))
    view = logic.filter_rows(rows, None if branch.startswith("All") else branch, None if segment.startswith("All") else segment)
    f3.caption(f"{len(view):,} customers match. Ranked by value at risk: probability of leaving in 90 days "
               f"times the 12-month value of the relationship.")

    k = logic.headline(view)
    m1, m2, m3 = st.columns(3)
    m1.metric("Customers to act on", f"{k['at_risk']:,}", help="Urgency 'act this week' or 'act this month'")
    m2.metric("Total value at risk", logic.usd(k["value_at_risk_usd"]))
    m3.metric(f"Held by the top {k['top_n']:,} customers", logic.pct(k["top_share"]),
              help=f"Share of value at risk in the top {int(100 * logic.TOP_SHARE)}% of the ranking: that is the call list")

    table = logic.worklist_table(view)
    st.caption("Select a row to open the customer's card.")
    event = st.dataframe(
        table, hide_index=True, use_container_width=True, height=330,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Risk in 90 days": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "Value at risk (USD)": st.column_config.NumberColumn(format="%.0f"),
            "Next action": st.column_config.TextColumn(width="large"),
        },
    )
    selected = event.selection.rows if event and event.selection else []
    if selected:
        open_card(str(table.iloc[selected[0]]["Customer"]))


# ----------------------------------------------------------------------------- 2. customer card
def card() -> None:
    h = st.session_state["health"]
    rows = api.worklist()
    refs = rows["customer_ref"].tolist()
    default = st.session_state.get("ref") or (refs[0] if refs else "")
    header("Customer card", snapshot_caption(h))

    c1, c2 = st.columns([2, 1])
    ref = c1.selectbox("Customer reference", refs, index=refs.index(default) if default in refs else 0,
                       help="Pseudonymised reference from the worklist")
    st.session_state["ref"] = ref
    if c2.button("Back to the worklist", use_container_width=True) and "worklist" in PAGES:
        st.switch_page(PAGES["worklist"])

    try:
        c = api.get(f"/customers/{ref}/card")
    except Exception:
        st.warning("No card for this reference at the latest snapshot.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk of leaving in 90 days", logic.pct(c["p_churn_90d"]))
    colour = URGENCY_COLOURS.get(c["urgency"], MUTED)
    m2.markdown(f"<div style='font-size:0.95rem;color:#5c6670;margin-bottom:0.35rem'>Urgency</div>"
                f"<div style='font-size:1.7rem;font-weight:600;color:{colour};line-height:1.2'>{c['urgency'].capitalize()}</div>",
                unsafe_allow_html=True)
    m3.metric("Value at risk", logic.usd(c["value_at_risk_usd"]))
    m4.metric("12-month value", logic.usd(c["value_12m_usd"]))
    st.caption(f"Rank {c['rank']} of {h['customers']:,}  |  {c['segment']} segment  |  {c['branch']} branch")

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Why Pulse flags this customer")
        if c["reasons"]:
            for i, r in enumerate(c["reasons"], 1):
                st.markdown(f"<span class='pulse-chip'>{r['family_label']}</span> **{i}.** {r['text']}", unsafe_allow_html=True)
        else:
            st.markdown("No single strong signal; the risk comes from many small ones.")
        st.subheader("Recommended action")
        st.markdown(f"**{c['action']}**")
        st.markdown(f"<div class='pulse-script'>{c['call_script']}</div>", unsafe_allow_html=True)
        nbp = c["next_best_product"]
        st.subheader("Next best product")
        st.markdown(f"**{nbp['product'].replace('_', ' ').capitalize()}**: {nbp['reason']}")
    with right:
        st.subheader("Last 12 months")
        try:
            hist = api.get(f"/customers/{ref}/history")
            long = logic.history_long(hist["points"])
        except Exception:
            long = pd.DataFrame(columns=["month", "series", "usd"])
        if long.empty:
            st.info("No monthly history available.")
        else:
            bal = long[long["series"] == "Month-end balance"]
            flows = long[long["series"] != "Month-end balance"]
            base = alt.Chart(bal).encode(x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y")))
            line = base.mark_line(color=ACCENT, strokeWidth=3, point=True).encode(
                y=alt.Y("usd:Q", title="Balance (USD)"), tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("usd:Q", format=",.0f", title="Balance")])
            bars = alt.Chart(flows).mark_bar(opacity=0.55).encode(
                x=alt.X("month:T", title=None), xOffset="series:N",
                y=alt.Y("usd:Q", title="Money in (USD)"),
                color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["Salary credit", "All money in"], range=["#B54708", MUTED]),
                                legend=alt.Legend(orient="top")),
                tooltip=["series:N", alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("usd:Q", format=",.0f")])
            st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent").properties(height=250), use_container_width=True)
            st.caption("Line: month-end balance. Bars: salary-sized credit and all money in, per month. The reasons on the left should be visible here.")


# ----------------------------------------------------------------------------- 3. insights
def insights() -> None:
    h = st.session_state["health"]
    header("What drives churn at the bank", snapshot_caption(h))

    try:
        d = api.get("/drivers")
    except Exception:
        d = None
    seg = api.get("/segments")["rows"]
    m = api.get("/metrics")

    a, b = st.columns([1, 1])
    with a:
        st.subheader("Top churn drivers")
        if d:
            df = logic.drivers_frame(d["families"])
            chart = alt.Chart(df).mark_bar(color=ACCENT).encode(
                x=alt.X("Share of model impact (%):Q", title="Share of model impact (%)"),
                y=alt.Y("Driver:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                tooltip=["Driver", "Share of model impact (%)", "Customers it pushes up (%)"]).properties(height=230)
            st.altair_chart(chart, use_container_width=True)
            for s in d["top_signals"][:4]:
                st.markdown(f"- **{s['family_label']}**: {s['reading'][0].upper() + s['reading'][1:]}.")
        else:
            st.info("Driver summary not available yet.")
    with b:
        st.subheader("Value at risk by group")
        dim = st.radio("Group by", list(logic.DIMENSION_LABELS), format_func=logic.DIMENSION_LABELS.get, horizontal=True, label_visibility="collapsed")
        t = logic.segment_table(seg, dim)
        chart = alt.Chart(t).mark_bar(color=ACCENT).encode(
            x=alt.X("Value at risk (USD):Q"), y=alt.Y("Group:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)),
            tooltip=["Group", "Customers", "Expected churn (%)", alt.Tooltip("Value at risk (USD):Q", format=",.0f"),
                     "Value at risk per customer (USD)"]).properties(height=230)
        st.altair_chart(chart, use_container_width=True)
        with st.expander("Table"):
            st.dataframe(t, hide_index=True, use_container_width=True,
                         column_config={"Value at risk (USD)": st.column_config.NumberColumn(format="%.0f"),
                                        "Value at risk per customer (USD)": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("Does the model beat simple rules, and is it fair?")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.caption("Held-out months after training. 'Churners caught in top 10%' is the share of actual leavers "
                   "that a call list of the riskiest 10% would have reached.")
        st.dataframe(logic.comparison_table(m["model_comparison_test"]), hide_index=True, use_container_width=True)
    with c2:
        st.caption("Protected attributes are never model inputs; this checks that flag and capture rates are similar across groups.")
        st.dataframe(logic.fairness_table(m["fairness_test"]), hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------- 4. ROI simulator
def roi() -> None:
    h = st.session_state["health"]
    header("What a retention campaign returns", snapshot_caption(h))
    rows = api.worklist()
    total = len(rows)

    s1, s2, s3, s4 = st.columns(4)
    top_n = s1.slider("Customers contacted", 0, total, min(500, total), step=max(1, total // 200))
    cost = s2.slider("Cost per contact (USD)", 0.0, 10.0, 2.0, step=0.5)
    incentive = s3.slider("Incentive per save (USD)", 0.0, 50.0, 10.0, step=1.0, help="Paid only for customers actually retained: fee waiver, rate bonus")
    save_rate_pct = s4.slider("Assumed save rate (%)", 0, 100, 30, step=5,
                              help="Share of would-be leavers that a contact retains. The bank supplies this; move it to test the case.")
    save_rate = save_rate_pct / 100

    sim = api.simulate(top_n, cost, incentive, save_rate)
    t, r = sim["targeted"], sim["random"]
    st.markdown("#### Pulse ranking versus random outreach, same budget")
    m = st.columns(4)
    m[0].metric("Revenue protected", logic.usd(t["revenue_protected_usd"]), delta=f"{t['revenue_protected_usd'] - r['revenue_protected_usd']:+,.0f} vs random")
    m[1].metric("Campaign cost", logic.usd(t["total_cost_usd"]), delta=f"{t['total_cost_usd'] - r['total_cost_usd']:+,.0f} vs random", delta_color="inverse")
    m[2].metric("Net return", logic.usd(t["net_return_usd"]), delta=f"{sim['uplift_net_usd']:+,.0f} vs random")
    m[3].metric("Expected customers saved", f"{t['expected_saves']:.0f}", delta=f"{t['expected_saves'] - r['expected_saves']:+.0f} vs random")

    curve = logic.roi_curve(rows["p_churn_90d"].to_numpy(), rows["value_12m_usd"].to_numpy(), cost, incentive, save_rate)
    best = logic.best_point(curve)
    lines = alt.Chart(curve).mark_line(strokeWidth=3).encode(
        x=alt.X("contacted:Q", title="Customers contacted"),
        y=alt.Y("net_return_usd:Q", title="Net return (USD)"),
        color=alt.Color("outreach:N", title=None, scale=alt.Scale(domain=["Pulse ranking", "Random outreach"], range=[ACCENT, MUTED]),
                        legend=alt.Legend(orient="top")),
        tooltip=["outreach:N", "contacted:Q", alt.Tooltip("net_return_usd:Q", format=",.0f", title="Net return (USD)")])
    marker = alt.Chart(pd.DataFrame({"contacted": [top_n]})).mark_rule(color="#B54708", strokeDash=[6, 4]).encode(x="contacted:Q")
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#CBD2D9").encode(y="y:Q")
    st.altair_chart(alt.layer(zero, lines, marker).properties(height=260), use_container_width=True)
    st.caption(f"Dashed line: the current setting. With these assumptions the Pulse line peaks at about "
               f"{best['contacted']:,} contacts ({logic.usd(best['net_return_usd'])} net). "
               f"Random outreach is the expected result of calling the same number of customers picked at random. "
               f"Expected values, not promises: the save rate is the assumption to agree with the bank.")
