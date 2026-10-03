# run via `streamlit run main.py`

import streamlit as st
from pipeline import DATASETS, load_data, train_model, predict_one

st.title("AML Transaction Classifier")


# Cached so re-running the script (every widget interaction) doesn't reload/retrain.
@st.cache_data(show_spinner="Loading datasets...")
def cached_load(name):
    return load_data(name)


@st.cache_resource(show_spinner=False)  # cache keyed on (name, epochs)
def cached_train(name, epochs):
    df = cached_load(name)
    bar = st.progress(0.0, text="Training...")

    def cb(ep, total, loss):
        bar.progress(ep / total, text=f"Epoch {ep}/{total} — loss {loss:.4f}")

    bundle = train_model(df, epochs=epochs, progress_cb=cb)
    bar.empty()
    return bundle


# dropdown tabs
dataset_name = st.selectbox("Choose a dataset to train on", list(DATASETS))
epochs = st.slider("Epochs", 1, 20, 5)

if dataset_name in ("HI-Medium", "LI-Medium"):
    st.warning("Medium is ~32M rows — first-time training may take several minutes.")

# train on click, not automatically, so a stray dropdown change doesn't kick off a long job.
if st.button("Train model"):
    bundle = cached_train(dataset_name, epochs)
    st.session_state["bundle"] = bundle  # stash it so the predict section can reach it
    m = bundle.metrics
    st.success(f"Trained on {dataset_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{m['precision']:.3f}")
    col2.metric("Recall", f"{m['recall']:.3f}")
    col3.metric("AUC-PR", f"{m['auc_pr']:.3f}")

# predicting transactions
if "bundle" in st.session_state:
    st.subheader("Test a transaction")
    amount = st.number_input("Amount", value=48500.0)
    pay_cur = st.selectbox("Payment currency", ["US Dollar", "Euro", "Yuan", "Shekel"])
    recv_cur = st.selectbox(
        "Receiving currency", ["US Dollar", "Euro", "Yuan", "Shekel"]
    )
    fmt = st.selectbox("Payment format", ["ACH", "Cheque", "Cash", "Credit Card"])

    if st.button("Predict"):
        prob = predict_one(
            st.session_state["bundle"],
            {
                "Timestamp": "2022-09-15 14:23:00",
                "Amount Received": amount,
                "Receiving Currency": recv_cur,
                "Amount Paid": amount,
                "Payment Currency": pay_cur,
                "Payment Format": fmt,
            },
        )
        st.metric("P(laundering)", f"{prob:.4f}")
        st.write("Flagged" if prob >= 0.5 else "Clean")
