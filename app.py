import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Spam Email Analysis",
    page_icon="📧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .clo-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📧 Spam Email Classification & Analysis</p>', unsafe_allow_html=True)

# Sidebar for file upload and settings
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload spam.csv file", type=['csv'])

# Alpha parameter for Laplace smoothing
alpha = st.sidebar.slider("Laplace Smoothing Alpha", 0.1, 10.0, 1.0, 0.1)

# Test size
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

# Random seed
random_seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

# Threshold for classification
threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)

# Global variables to store data
@st.cache_data
def load_data(file):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file)
    df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
    return df

@st.cache_data
def prepare_data(df, test_size, random_seed):
    """Prepare train/test split and vectorize"""
    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], df['label'], 
        test_size=test_size, 
        random_state=random_seed, 
        stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

def train_naive_bayes(X_train, y_train, alpha):
    """Train Naive Bayes model and return components"""
    # Bag-of-words counts
    vect = CountVectorizer(ngram_range=(1,1), min_df=1)
    X_train_counts = vect.fit_transform(X_train)
    
    # Estimate priors
    N = len(y_train)
    N_S = y_train.sum()
    N_H = N - N_S
    P_S = N_S / N
    P_H = N_H / N
    
    # Class token counts with Laplace smoothing
    spam_mask = (y_train == 1).to_numpy().nonzero()[0]
    ham_mask = (y_train == 0).to_numpy().nonzero()[0]
    
    spam_counts = X_train_counts[spam_mask].sum(axis=0) + alpha
    ham_counts = X_train_counts[ham_mask].sum(axis=0) + alpha
    spam_total = spam_counts.sum()
    ham_total = ham_counts.sum()
    
    log_pw_spam = np.log(spam_counts / spam_total).A1
    log_pw_ham = np.log(ham_counts / ham_total).A1
    
    return vect, X_train_counts, log_pw_spam, log_pw_ham, P_S, P_H

def posterior_prob(row_counts, log_pw_spam, log_pw_ham, P_S, P_H):
    """Compute posterior probability for a sparse row"""
    ll_spam = row_counts.dot(log_pw_spam).item()
    ll_ham = row_counts.dot(log_pw_ham).item()
    
    s_spam = ll_spam + np.log(P_S)
    s_ham = ll_ham + np.log(P_H)
    
    maxs = max(s_spam, s_ham)
    num = np.exp(s_spam - maxs)
    den = np.exp(s_spam - maxs) + np.exp(s_ham - maxs)
    return float(num / den)

# Main app logic
if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    # Display dataset info
    st.success(f"✅ Dataset loaded successfully! Total emails: {len(df)}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Emails", len(df))
    with col2:
        st.metric("Ham Emails", (df['label'] == 0).sum())
    with col3:
        st.metric("Spam Emails", (df['label'] == 1).sum())
    with col4:
        st.metric("Spam %", f"{(df['label'].sum() / len(df) * 100):.2f}%")
    
    # Show sample data
    with st.expander("📊 View Sample Data"):
        st.dataframe(df.head(10))
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, test_size, random_seed)
    
    # Train model
    with st.spinner("Training Naive Bayes model..."):
        vect, X_train_counts, log_pw_spam, log_pw_ham, P_S, P_H = train_naive_bayes(X_train, y_train, alpha)
        X_test_counts = vect.transform(X_test)
        X_counts_all = vect.transform(df['Message'])
    
    st.success("✅ Model trained successfully!")
    
    # Tabs for different CLOs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 CLO 1: Bayes Theorem",
        "📈 CLO 2: Continuous Variables",
        "📉 CLO 3: Beta Distribution",
        "🎯 CLO 4: False Positive Rate",
        "🔄 CLO 5: Markov Chains",
        "🤖 Test Model"
    ])
    
    # CLO 1: Bayes Theorem
    with tab1:
        st.markdown('<p class="clo-header">CLO 1: Calculate Probability Using Bayes Theorem</p>', unsafe_allow_html=True)
        
        st.write("### Model Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("P(Spam) - Prior", f"{P_S:.4f}")
            st.metric("P(Ham) - Prior", f"{P_H:.4f}")
        with col2:
            st.metric("Vocabulary Size", len(vect.get_feature_names_out()))
            st.metric("Laplace Alpha", alpha)
        
        st.write("### Test Sample Predictions")
        n_samples = st.slider("Number of test samples to show", 5, 50, 10)
        
        results = []
        for i in range(min(n_samples, len(X_test))):
            p_spam = posterior_prob(X_test_counts[i], log_pw_spam, log_pw_ham, P_S, P_H)
            true_label = "Spam" if y_test.iloc[i] == 1 else "Ham"
            pred_label = "Spam" if p_spam >= threshold else "Ham"
            
            results.append({
                "Index": i,
                "Message": X_test.iloc[i][:100] + "...",
                "True Label": true_label,
                "P(Spam|x)": f"{p_spam:.6f}",
                "Predicted": pred_label,
                "Correct": "✅" if true_label == pred_label else "❌"
            })
        
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    # CLO 2: Continuous Random Variables
    with tab2:
        st.markdown('<p class="clo-header">CLO 2: Spam Scores as Continuous Random Variables</p>', unsafe_allow_html=True)
        
        # Compute posteriors for all messages
        with st.spinner("Computing posterior probabilities..."):
            posteriors = [posterior_prob(X_counts_all[i], log_pw_spam, log_pw_ham, P_S, P_H) 
                         for i in range(X_counts_all.shape[0])]
            post = np.array(posteriors)
        
        post_spam = post[df['label'] == 1]
        post_ham = post[df['label'] == 0]
        
        # Overall statistics
        st.write("### Overall Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Posterior Score", f"{post.mean():.6f}")
        with col2:
            st.metric("Variance", f"{post.var(ddof=1):.6f}")
        
        # Class-conditional statistics
        st.write("### Class-Conditional Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Spam Messages**")
            st.metric("Mean | Spam", f"{post_spam.mean():.6f}")
            st.metric("Variance | Spam", f"{post_spam.var(ddof=1):.6f}")
        with col2:
            st.write("**Ham Messages**")
            st.metric("Mean | Ham", f"{post_ham.mean():.6f}")
            st.metric("Variance | Ham", f"{post_ham.var(ddof=1):.6f}")
        
        # Visualization
        st.write("### Distribution Visualization")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(post_ham, bins=50, alpha=0.6, label='Ham', color='green')
        axes[0].hist(post_spam, bins=50, alpha=0.6, label='Spam', color='red')
        axes[0].set_xlabel('Posterior Probability P(Spam|x)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Posterior Probabilities')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        data_box = pd.DataFrame({
            'Probability': np.concatenate([post_ham, post_spam]),
            'Class': ['Ham']*len(post_ham) + ['Spam']*len(post_spam)
        })
        sns.boxplot(data=data_box, x='Class', y='Probability', ax=axes[1])
        axes[1].set_title('Box Plot by Class')
        axes[1].grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # CLO 3: Beta Distribution
    with tab3:
        st.markdown('<p class="clo-header">CLO 3: Beta Distribution for Classification Confidence</p>', unsafe_allow_html=True)
        
        # Clip posteriors
        epsilon = 1e-6
        post_safe = np.clip(post, epsilon, 1 - epsilon)
        post_spam_safe = np.clip(post_spam, epsilon, 1 - epsilon)
        post_ham_safe = np.clip(post_ham, epsilon, 1 - epsilon)
        
        # Method of Moments fit
        st.write("### Overall Posterior Beta Fit")
        sbar = post_safe.mean()
        v = post_safe.var(ddof=1)
        
        if v >= sbar * (1 - sbar):
            st.info("Variance too large for Method of Moments. Using MLE.")
            a_mle, b_mle, loc, scale = stats.beta.fit(post_safe, floc=0, fscale=1)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Alpha (MLE)", f"{a_mle:.6f}")
            with col2:
                st.metric("Beta (MLE)", f"{b_mle:.6f}")
        else:
            t = (sbar * (1 - sbar) / v) - 1
            alpha_hat = sbar * t
            beta_hat = (1 - sbar) * t
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Alpha (MoM)", f"{alpha_hat:.6f}")
            with col2:
                st.metric("Beta (MoM)", f"{beta_hat:.6f}")
        
        # Class-conditional Beta fit
        st.write("### Class-Conditional Beta Parameters")
        a_spam, b_spam, _, _ = stats.beta.fit(post_spam_safe, floc=0, fscale=1)
        a_ham, b_ham, _, _ = stats.beta.fit(post_ham_safe, floc=0, fscale=1)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Spam Beta Parameters**")
            st.metric("Alpha", f"{a_spam:.6f}")
            st.metric("Beta", f"{b_spam:.6f}")
            mean_spam = a_spam / (a_spam + b_spam)
            var_spam = (a_spam * b_spam) / ((a_spam + b_spam)**2 * (a_spam + b_spam + 1))
            st.metric("Mean", f"{mean_spam:.6f}")
            st.metric("Variance", f"{var_spam:.6f}")
        
        with col2:
            st.write("**Ham Beta Parameters**")
            st.metric("Alpha", f"{a_ham:.6f}")
            st.metric("Beta", f"{b_ham:.6f}")
            mean_ham = a_ham / (a_ham + b_ham)
            var_ham = (a_ham * b_ham) / ((a_ham + b_ham)**2 * (a_ham + b_ham + 1))
            st.metric("Mean", f"{mean_ham:.6f}")
            st.metric("Variance", f"{var_ham:.6f}")
        
        # Visualization
        st.write("### Beta Distribution Fit Visualization")
        x = np.linspace(0, 1, 1000)
        
        fig = go.Figure()
        
        # Spam Beta
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, a_spam, b_spam),
            mode='lines', name=f'Spam Beta({a_spam:.2f}, {b_spam:.2f})',
            line=dict(color='red', width=2)
        ))
        
        # Ham Beta
        fig.add_trace(go.Scatter(
            x=x, y=stats.beta.pdf(x, a_ham, b_ham),
            mode='lines', name=f'Ham Beta({a_ham:.2f}, {b_ham:.2f})',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Beta Distribution Fits for Spam and Ham',
            xaxis_title='Posterior Probability',
            yaxis_title='Density',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # CLO 4: False Positive Rate
    with tab4:
        st.markdown('<p class="clo-header">CLO 4: False Positive Rate Analysis</p>', unsafe_allow_html=True)
        
        # Compute predictions
        y_pred = [1 if posterior_prob(X_test_counts[i], log_pw_spam, log_pw_ham, P_S, P_H) >= threshold 
                  else 0 for i in range(X_test_counts.shape[0])]
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        st.write("### Confusion Matrix")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            conf_matrix = pd.DataFrame(
                [[tn, fp], [fn, tp]],
                columns=['Predicted Ham', 'Predicted Spam'],
                index=['Actual Ham', 'Actual Spam']
            )
            st.dataframe(conf_matrix, use_container_width=True)
        
        # Metrics
        st.write("### Performance Metrics")
        N_H_test = tn + fp
        fpr = fp / N_H_test if N_H_test > 0 else 0
        var_binom = fpr * (1 - fpr) / N_H_test if N_H_test > 0 else 0
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1 Score", f"{f1:.4f}")
        
        st.write("### False Positive Rate Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("False Positive Rate", f"{fpr:.6f}")
            st.metric("False Positives", fp)
            st.metric("True Negatives", tn)
        with col2:
            st.metric("Binomial Variance", f"{var_binom:.9f}")
            st.metric("Standard Error", f"{np.sqrt(var_binom):.6f}")
        
        # K-Fold Cross-Validation
        st.write("### K-Fold Cross-Validation Analysis")
        k = st.slider("Number of folds", 2, 10, 5)
        
        with st.spinner(f"Running {k}-fold cross-validation..."):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
            fprs = []
            accuracies = []
            
            for train_idx, test_idx in skf.split(df['Message'], df['label']):
                vect_cv = CountVectorizer(ngram_range=(1,1), min_df=1)
                Xtr = vect_cv.fit_transform(df['Message'].iloc[train_idx])
                Xte = vect_cv.transform(df['Message'].iloc[test_idx])
                ytr = df['label'].iloc[train_idx].values
                yte = df['label'].iloc[test_idx].values
                
                mnb = MultinomialNB(alpha=alpha)
                mnb.fit(Xtr, ytr)
                ypred = mnb.predict(Xte)
                
                tn_cv, fp_cv, fn_cv, tp_cv = confusion_matrix(yte, ypred).ravel()
                fprs.append(fp_cv / (fp_cv + tn_cv) if (fp_cv + tn_cv) > 0 else 0)
                accuracies.append(accuracy_score(yte, ypred))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean FPR across folds", f"{np.mean(fprs):.6f}")
            st.metric("Empirical Variance", f"{np.var(fprs, ddof=1):.9f}")
        with col2:
            st.metric("Mean Accuracy", f"{np.mean(accuracies):.4f}")
            st.metric("Std Dev Accuracy", f"{np.std(accuracies, ddof=1):.4f}")
        
        # Plot FPRs
        fig = px.line(x=range(1, k+1), y=fprs, markers=True,
                      labels={'x': 'Fold', 'y': 'False Positive Rate'},
                      title='False Positive Rate Across Folds')
        fig.add_hline(y=np.mean(fprs), line_dash="dash", 
                      annotation_text=f"Mean: {np.mean(fprs):.4f}")
        st.plotly_chart(fig, use_container_width=True)
    
    # CLO 5: Markov Chains
    with tab5:
        st.markdown('<p class="clo-header">CLO 5: Spam Pattern Evolution with Markov Chains</p>', unsafe_allow_html=True)
        
        st.info("⚠️ Note: This analysis assumes the dataset order represents temporal sequence.")
        
        labels = df['label'].values
        
        # Count transitions
        trans = np.zeros((2, 2), dtype=int)
        for i in range(len(labels) - 1):
            trans[labels[i], labels[i+1]] += 1
        
        # Transition probabilities
        row_sums = trans.sum(axis=1).astype(float)
        P = np.zeros_like(trans, dtype=float)
        for i in [0, 1]:
            if row_sums[i] > 0:
                P[i, :] = trans[i, :] / row_sums[i]
        
        st.write("### Transition Count Matrix")
        trans_df = pd.DataFrame(
            trans,
            columns=['To Ham', 'To Spam'],
            index=['From Ham', 'From Spam']
        )
        st.dataframe(trans_df, use_container_width=True)
        
        st.write("### Transition Probability Matrix")
        prob_df = pd.DataFrame(
            P,
            columns=['To Ham', 'To Spam'],
            index=['From Ham', 'From Spam']
        )
        st.dataframe(prob_df.style.format("{:.6f}"), use_container_width=True)
        
        # Stationary distribution
        p_HS = P[0, 1]
        p_SH = P[1, 0]
        
        if (p_SH + p_HS) > 0:
            pi_h = p_SH / (p_SH + p_HS)
            pi_s = p_HS / (p_SH + p_HS)
        else:
            pi_h = pi_s = np.nan
        
        st.write("### Markov Chain Properties")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Stationary P(Ham)", f"{pi_h:.6f}")
            st.metric("P(Ham → Spam)", f"{p_HS:.6f}")
        with col2:
            st.metric("Stationary P(Spam)", f"{pi_s:.6f}")
            st.metric("P(Spam → Ham)", f"{p_SH:.6f}")
        
        # Expected time to reach spam
        if p_HS > 0:
            exp_time = 1.0 / p_HS
        else:
            exp_time = np.inf
        
        st.metric("Expected Steps to Reach Spam from Ham", f"{exp_time:.2f}")
        
        # Visualize transition matrix
        st.write("### Transition Probability Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(P, annot=True, fmt='.4f', cmap='YlOrRd', 
                   xticklabels=['Ham', 'Spam'],
                   yticklabels=['Ham', 'Spam'],
                   ax=ax, cbar_kws={'label': 'Probability'})
        ax.set_title('State Transition Probability Matrix')
        ax.set_xlabel('Next State')
        ax.set_ylabel('Current State')
        st.pyplot(fig)
        
        # Simulate chain
        st.write("### Markov Chain Simulation")
        n_steps = st.slider("Number of simulation steps", 10, 200, 50)
        initial_state = st.radio("Initial State", ["Ham", "Spam"])
        
        if st.button("Run Simulation"):
            state = 0 if initial_state == "Ham" else 1
            states = [state]
            
            for _ in range(n_steps):
                state = np.random.choice([0, 1], p=P[state])
                states.append(state)
            
            fig = px.line(x=range(len(states)), y=states,
                         labels={'x': 'Step', 'y': 'State'},
                         title=f'Markov Chain Simulation ({n_steps} steps)')
            fig.update_yaxes(ticktext=['Ham', 'Spam'], tickvals=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            spam_proportion = np.mean(states)
            st.metric("Proportion of Spam States", f"{spam_proportion:.4f}")
    
    # Test Model Tab
    with tab6:
        st.markdown('<p class="clo-header">🤖 Test the Model</p>', unsafe_allow_html=True)
        
        st.write("Enter an email message to classify:")
        
        user_input = st.text_area("Email Message", height=150, 
                                  placeholder="Type or paste your email message here...")
        
        if st.button("Classify Message", type="primary"):
            if user_input.strip():
                # Transform input
                input_counts = vect.transform([user_input])
                
                # Get probability
                prob_spam = posterior_prob(input_counts[0], log_pw_spam, log_pw_ham, P_S, P_H)
                prob_ham = 1 - prob_spam
                
                # Display result
                st.write("### Classification Result")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if prob_spam >= threshold:
                        st.error("🚫 SPAM")
                    else:
                        st.success("✅ HAM")
                
                with col2:
                    st.metric("Spam Probability", f"{prob_spam:.4f}")
                
                with col3:
                    st.metric("Ham Probability", f"{prob_ham:.4f}")
                
                # Probability bar
                fig = go.Figure(go.Bar(
                    x=[prob_ham, prob_spam],
                    y=['Ham', 'Spam'],
                    orientation='h',
                    marker=dict(color=['green', 'red'])
                ))
                fig.update_layout(
                    title='Classification Probabilities',
                    xaxis_title='Probability',
                    showlegend=False,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top features
                st.write("### Message Analysis")
                words = user_input.lower().split()
                vocab = vect.get_feature_names_out()
                
                word_probs = []
                for word in set(words):
                    if word in vocab:
                        idx = np.where(vocab == word)[0][0]
                        spam_score = np.exp(log_pw_spam[idx])
                        ham_score = np.exp(log_pw_ham[idx])
                        word_probs.append({
                            'Word': word,
                            'P(word|spam)': spam_score,
                            'P(word|ham)': ham_score,
                            'Spam/Ham Ratio': spam_score / ham_score if ham_score > 0 else np.inf
                        })
                
                if word_probs:
                    word_df = pd.DataFrame(word_probs).sort_values('Spam/Ham Ratio', ascending=False)
                    st.dataframe(word_df.head(10), use_container_width=True)
            else:
                st.warning("Please enter a message to classify.")

else:
    st.info("👈 Please upload the spam.csv file from the sidebar to begin analysis.")
    
    st.markdown("""
    ### 📋 Instructions
    
    1. **Download the dataset**: Get the spam.csv file from Kaggle
       - Link: https://www.kaggle.com/datasets/venky73/spam-mails-dataset
    
    2. **Upload the file** using the sidebar uploader
    
    3. **Explore the CLOs**:
       - **CLO 1**: Calculate spam probability using Bayes theorem
       - **CLO 2**: Analyze spam scores as continuous random variables
       - **CLO 3**: Model classification confidence with Beta distribution
       - **CLO 4**: Calculate and analyze false positive rates
       - **CLO 5**: Model spam pattern evolution using Markov chains
       - **Test Model**: Try classifying your own email messages!
    
    ### ⚙️ Configuration Options
    
    - Adjust Laplace smoothing parameter
    - Change train/test split ratio
    - Modify classification threshold
    - Set random seed for reproducibility
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📧 Spam Email Classification System | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)