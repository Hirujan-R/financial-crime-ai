"""Analyst Dashboard for the AI-Powered Financial Crime Investigation Platform.

Run with:  streamlit run src/financial_crime_ai/dashboard/app.py

The analyst sees the investigation queue (future transactions scored by the
locked IBM AML model), clicks a flagged transaction and drills into the full
AI case report: why it was flagged, matched typologies, the transaction
network, historical behaviour, similar cases, supporting evidence, model
confidence and the recommended next action.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from financial_crime_ai.agent.context import InvestigatorContext  # noqa: E402
from financial_crime_ai.agent.investigator import CaseInvestigator  # noqa: E402

st.set_page_config(
    page_title="Financial Crime Investigation",
    page_icon="🕵️",
    layout="wide",
)

st.title("AI-Powered Financial Crime Investigation Platform")


@st.cache_resource
def load_context() -> tuple[InvestigatorContext, pd.DataFrame]:
    ctx = InvestigatorContext.from_project_dir(ROOT)
    queue = pd.read_parquet(
        ROOT / "data" / "06_reporting" / "investigation_queue.parquet"
    )
    return ctx, queue


@st.cache_resource
def load_investigator(_ctx: InvestigatorContext) -> CaseInvestigator:
    return CaseInvestigator(_ctx)


ctx, queue = load_context()
investigator = load_investigator(ctx)

st.sidebar.header("Platform status")
st.sidebar.caption(
    f"Investigation engine: **{investigator.mode}**\n\n"
    "Set `OPENAI_API_KEY` to enable the LLM agent with structured tool calling; "
    "otherwise the deterministic engine produces the same report."
)
n_critical = int((queue["priority"] == "critical").sum())
n_high = int((queue["priority"] == "high").sum())
n_med = int((queue["priority"] == "medium").sum())
st.sidebar.metric("Open cases", len(queue))
st.sidebar.metric("Critical", n_critical)
st.sidebar.metric("High", n_high)
st.sidebar.metric("Medium", n_med)

if not queue.empty:
    st.sidebar.divider()
    st.sidebar.subheader("Test-set model metrics")
    tm = ctx.test_metrics.set_index("metric")["value"].to_dict()
    st.sidebar.metric("Test AUC", f"{tm.get('auc', 0):.4f}")
    st.sidebar.metric("Test Avg Precision", f"{tm.get('average_precision', 0):.4f}")
    st.sidebar.metric(
        "Test Recall@threshold", f"{tm.get('recall_at_threshold', 0):.2%}"
    )


def risk_bar(score: float, width: int = 18) -> str:
    filled = int(round(score * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"`{bar} {score:.2f}`"


# ---------------------------------------------------------------------------
# Investigation queue
# ---------------------------------------------------------------------------
st.subheader("Investigation queue")
if queue.empty:
    st.warning("No flagged cases above the model threshold.")
else:
    view = queue[
        [
            "txn_id",
            "priority",
            "risk_score",
            "account_id",
            "receiver_account_id",
            "amount",
            "timestamp",
            "alert_type",
        ]
    ].copy()
    view["risk"] = view["risk_score"].apply(risk_bar)
    display = view.rename(
        columns={
            "txn_id": "Transaction #",
            "priority": "Priority",
            "risk": "Risk",
            "account_id": "Sender account",
            "receiver_account_id": "Receiver account",
            "amount": "Amount",
            "timestamp": "Time bucket",
            "alert_type": "Pattern label",
        }
    )
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "Amount": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    st.divider()

    # -----------------------------------------------------------------------
    # Case selection
    # -----------------------------------------------------------------------
    selected = st.selectbox(
        "Open a case:",
        queue["txn_id"].astype(str).tolist(),
        format_func=lambda t: (
            f"{t}  ·  sender {queue.loc[queue['txn_id'].astype(str) == t, 'account_id'].iloc[0]}  "
            f"·  risk {queue.loc[queue['txn_id'].astype(str) == t, 'risk_score'].iloc[0]:.2f}"
        ),
    )
    row = queue[queue["txn_id"].astype(str) == selected].iloc[0]

    st.header(
        f"Case {selected} — sender {row['account_id']} "
        f"({row['priority']}, risk {row['risk_score']:.2f})"
    )

    with st.spinner("AI investigator assembling case evidence..."):
        report = investigator.investigate(selected)

    # -----------------------------------------------------------------------
    # Case report
    # -----------------------------------------------------------------------
    (
        tab_overview,
        tab_why,
        tab_network,
        tab_history,
        tab_similar,
        tab_evidence,
        tab_actions,
    ) = st.tabs(
        [
            "Overview",
            "Why flagged",
            "Connected accounts",
            "History",
            "Similar cases",
            "Evidence",
            "Recommended action",
        ]
    )

    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk score", f"{report.risk_score:.3f}")
        c2.metric("Model confidence", f"{report.model_confidence:.0%}")
        c3.metric("Generated by", report.generated_by.replace("-", " ").title())
        st.markdown("#### AI summary")
        st.info(report.summary)
        st.download_button(
            "Download case report (JSON)",
            data=json.dumps(report.to_dict(), indent=2),
            file_name=f"case_{report.txn_id}.json",
            mime="application/json",
        )
        st.markdown("#### Case facts")
        hist = report.historical_behaviour
        if hist:
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Behaviour profile", hist.get("behavior_id", "—"))
            f2.metric("Transactions", hist.get("n_transactions", "—"))
            f3.metric("Total sent", f"${hist.get('total_sent', 0):,.0f}")
            f4.metric("Total received", f"${hist.get('total_received', 0):,.0f}")
            st.write(
                f"Outgoing: **{hist.get('n_outgoing', 0)}** · "
                f"Incoming: **{hist.get('n_incoming', 0)}** · "
                f"Avg amount: **${hist.get('avg_amount', 0):,.0f}** · "
                f"Max amount: **${hist.get('max_amount', 0):,.0f}** · "
                f"Prior alert involvement: **{hist.get('in_known_alert', False)}**"
            )

    with tab_why:
        st.markdown("#### Why this was flagged")
        for w in report.why_flagged:
            st.markdown(f"- {w}")
        if not report.why_flagged:
            st.warning("No plain-language drivers found for this case.")
        st.markdown("#### Matched typologies")
        for t in report.matched_typologies:
            st.markdown(f"- **{t}**")
        st.markdown("#### Model contributions")
        contrib_df = pd.DataFrame(
            [
                {
                    "Feature": c.label,
                    "Value": round(c.value, 3),
                    "Contribution": round(c.contribution, 4),
                }
                for c in report.top_contributors
            ]
        )
        st.dataframe(contrib_df, width="stretch", hide_index=True)

    with tab_network:
        st.markdown("#### Counterparty network")
        net = pd.DataFrame(report.connected_accounts)
        if not net.empty:
            net_disp = net.copy()
            net_disp["value"] = net_disp["value"].apply(lambda v: f"${v:,.0f}")
            st.dataframe(net_disp, width="stretch", hide_index=True)
        else:
            st.info("No counterparty edges recorded for this account.")

    with tab_history:
        st.markdown("#### Recent account activity")
        recent = report.historical_behaviour.get("recent", [])
        if recent:
            hist_df = pd.DataFrame(recent)
            hist_df["TX_AMOUNT"] = hist_df["TX_AMOUNT"].apply(lambda v: f"${v:,.2f}")
            st.dataframe(
                hist_df[
                    [
                        "TX_ID",
                        "TIMESTAMP",
                        "direction",
                        "SENDER_ACCOUNT_ID",
                        "RECEIVER_ACCOUNT_ID",
                        "TX_AMOUNT",
                        "IS_FRAUD",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No recent transactions available.")

    with tab_similar:
        st.markdown("#### Similar accounts")
        sim = pd.DataFrame(report.similar_cases)
        if not sim.empty:
            sim_disp = sim.copy()
            sim_disp["evidence"] = sim_disp["evidence"].apply(
                lambda e: (
                    f"flow ${e.get('total_flow', 0):,.0f} · "
                    f"txns {e.get('n_transactions', 0)} · "
                    f"pattern={e.get('pattern', 'n/a')}"
                    if e
                    else ""
                )
            )
            st.dataframe(
                sim_disp[["account_id", "risk_score", "similarity", "evidence"]],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No similar accounts found.")

    with tab_evidence:
        st.markdown("#### Supporting evidence")
        for e in report.supporting_evidence:
            st.markdown(f"- {e}")
        st.markdown("#### Prior alert involvement")
        if report.sanctions_matches:
            for s in report.sanctions_matches:
                st.error(f"- {s}")
        else:
            st.success("No prior alert involvement for this account.")

    with tab_actions:
        st.markdown("#### Recommended next action")
        st.success(report.recommended_action)
        st.caption(f"Action confidence: **{report.action_confidence}**")
        st.markdown("#### Suggested escalation options")
        st.button("Freeze account", type="primary")
        st.button("File SAR / STR")
        st.button("Enhance monitoring (30d)")
        st.button("Close case")
