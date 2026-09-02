"""AMLSim-style controlled scenario generator.

Emits synthetic transaction networks in the IBM AMLSim schema with *known*
injected typologies so we can test whether a model generalises to patterns
it never saw during training:

Training domain (known):   normal + structuring + layering
Held-out domain (novel):   normal + cycle (circular) + fan_out + fan_in

Each transaction carries an ``ALERT_TYPE`` so the per-typology evaluation in
the ``aml_sim`` pipeline can quantify generalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

THRESHOLD = 10_000.0


def generate_amlsim_data(
    n_accounts: int,
    n_timestamps: int,
    seed: int,
    normal_rate: float,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    accounts = _make_accounts(n_accounts, rng)
    txns: list[dict] = []
    counter = 0

    def emit(sender, receiver, amount, ts, is_fraud, alert_type):
        nonlocal counter
        counter += 1
        txns.append(
            {
                "TX_ID": counter,
                "SENDER_ACCOUNT_ID": int(sender),
                "RECEIVER_ACCOUNT_ID": int(receiver),
                "TX_TYPE": "TRANSFER",
                "TX_AMOUNT": float(amount),
                "TIMESTAMP": int(ts),
                "IS_FRAUD": bool(is_fraud),
                "ALERT_TYPE": alert_type,
            }
        )

    def account(prefix: str, i: int) -> int:
        return int(f"{ord(prefix) * 1_000_000 + i}")

    # ---- normal background traffic ----------------------------------------
    n_ts = n_timestamps
    for ts in range(n_ts):
        n_tx = int(rng.poisson(normal_rate * n_accounts))
        senders = rng.integers(0, n_accounts, size=n_tx)
        receivers = rng.integers(0, n_accounts, size=n_tx)
        amounts = rng.lognormal(np.log(500.0), 0.8, size=n_tx)
        for s, r, a in zip(senders, receivers, amounts):
            if s == r:
                continue
            emit(s, r, a, ts, False, "none")

    # ---- structuring: many deposits just under the threshold, consolidated -
    n_sinks = 40
    for k in range(n_sinks):
        sink = account("S", k)
        n_dep = int(rng.integers(12, 20))
        start_ts = int(rng.integers(0, n_ts - 4))
        for i in range(n_dep):
            sender = account("D", k * 100 + i)
            emit(
                sender,
                sink,
                float(rng.uniform(9_400, 9_999)),
                start_ts,
                True,
                "structuring",
            )
        emit(
            sink,
            account("X", k),
            float(n_dep * 9_700),
            start_ts + 1,
            True,
            "structuring",
        )

    # ---- layering: mule chains pass funds quickly then exit -----------------
    for k in range(30):
        chain = [account("L", k * 6 + j) for j in range(4)]
        start_ts = int(rng.integers(0, n_ts - 6))
        amt = float(rng.uniform(5_000, 80_000))
        for j in range(len(chain) - 1):
            emit(chain[j], chain[j + 1], amt, start_ts + j, True, "layering")
        emit(chain[-1], account("Y", k), amt, start_ts + len(chain), True, "layering")

    # ---- cycle: circular transfers A->B->C->A ------------------------------
    for k in range(30):
        a, b, c = account("C", k * 3), account("C", k * 3 + 1), account("C", k * 3 + 2)
        ts0 = int(rng.integers(0, n_ts - 3))
        amt = float(rng.uniform(5_000, 60_000))
        emit(a, b, amt, ts0, True, "cycle")
        emit(b, c, amt, ts0 + 1, True, "cycle")
        emit(c, a, amt, ts0 + 2, True, "cycle")

    # ---- fan_out: one source disperses to many receivers --------------------
    for k in range(25):
        src = account("O", k)
        ts0 = int(rng.integers(0, n_ts - 2))
        for i in range(int(rng.integers(8, 14))):
            emit(
                src,
                account("O", k * 100 + i),
                float(rng.uniform(1_000, 8_000)),
                ts0,
                True,
                "fan_out",
            )

    # ---- fan_in: many senders concentrate into one sink ---------------------
    for k in range(25):
        sink = account("I", k)
        ts0 = int(rng.integers(0, n_ts - 2))
        for i in range(int(rng.integers(8, 14))):
            emit(
                account("I", k * 100 + i),
                sink,
                float(rng.uniform(1_000, 8_000)),
                ts0,
                True,
                "fan_in",
            )

    txn = pd.DataFrame(txns).sort_values("TIMESTAMP").reset_index(drop=True)

    # Split by typology: train sees normal + structuring + layering;
    # the held-out test sees normal + cycle + fan_out + fan_in.
    known = {"structuring", "layering"}
    train = txn[txn["ALERT_TYPE"].isin(known | {"none"})]
    test = txn[~txn["ALERT_TYPE"].isin(known)]

    return {
        "accounts": accounts,
        "train_transactions": train,
        "test_transactions": test,
        "summary": pd.DataFrame(
            [
                {
                    "alert_type": a,
                    "n": int(txn[txn["ALERT_TYPE"] == a].shape[0]),
                    "domain": "known" if a in known else "held-out",
                }
                for a in txn["ALERT_TYPE"].value_counts().index
            ]
        ),
    }


def _make_accounts(n_accounts: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ACCOUNT_ID": np.arange(n_accounts),
            "CUSTOMER_ID": [f"C_{i}" for i in range(n_accounts)],
            "INIT_BALANCE": rng.lognormal(np.log(280.0), 0.4, size=n_accounts),
            "COUNTRY": "US",
            "ACCOUNT_TYPE": "I",
            "IS_FRAUD": False,
            "TX_BEHAVIOR_ID": rng.integers(1, 6, size=n_accounts),
        }
    )
