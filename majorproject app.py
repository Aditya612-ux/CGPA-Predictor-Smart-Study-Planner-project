import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ----------------- App Configuration -----------------
st.set_page_config(page_title="CGPA Predictor & Study Planner", page_icon="🎓", layout="wide")

st.title("🎓 CGPA Predictor & Smart Study Planner")
st.markdown("Predict your expected CGPA, view past trends, and generate a dynamic study timetable.")

# ----------------- ML Data Generation & Training -----------------
@st.cache_resource
def train_model():
    # We generate synthetic student data to train our model
    np.random.seed(42)
    n_samples = 200
    
    # Synthetic Features
    prev_cgpa = np.random.uniform(5.0, 10.0, n_samples)
    study_hours = np.random.uniform(1.0, 10.0, n_samples)
    attendance = np.random.uniform(50.0, 100.0, n_samples)
    assignments = np.random.uniform(40.0, 100.0, n_samples)
    
    # Target linearly dependent on the features + noise
    expected_cgpa = (0.5 * prev_cgpa + 
                     0.1 * (study_hours / 10) * 10 + 
                     0.2 * (attendance / 100) * 10 + 
                     0.2 * (assignments / 100) * 10)
    
    # Clip between 0 and 10 mapped scale
    expected_cgpa = np.clip(expected_cgpa + np.random.normal(0, 0.2, n_samples), 0, 10)
    
    X = pd.DataFrame({
        'Previous CGPA': prev_cgpa,
        'Study Hours': study_hours,
        'Attendance': attendance,
        'Assignments': assignments
    })
    y = expected_cgpa
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return scaler, model

scaler, model = train_model()

# ----------------- Sidebar: User Inputs -----------------
st.sidebar.header("📊 Your Current Stats")

prev_cgpa_in = st.sidebar.slider("Previous CGPA", 0.0, 10.0, 7.5, 0.1)
study_horas_in = st.sidebar.slider("Daily Study Hours", 0.0, 15.0, 3.0, 0.5)
attendance_in = st.sidebar.slider("Attendance (%)", 0.0, 100.0, 80.0, 1.0)
assignments_in = st.sidebar.slider("Assignments Completed (%)", 0.0, 100.0, 85.0, 1.0)

st.sidebar.markdown("---")
target_cgpa = st.sidebar.number_input("🎯 Target CGPA", min_value=0.0, max_value=10.0, value=8.5, step=0.1)


# ----------------- Tabs -----------------
tab1, tab2, tab3 = st.tabs(["🔮 CGPA Prediction", "📈 Progress Analysis", "📅 Study Timetable"])

# ================= TAB 1: Prediction =================
with tab1:
    st.header("Predict Your Next Semester CGPA")
    
    # Run user input through the prediction model
    user_data = pd.DataFrame({
        'Previous CGPA': [prev_cgpa_in],
        'Study Hours': [study_horas_in],
        'Attendance': [attendance_in],
        'Assignments': [assignments_in]
    })
    
    user_data_scaled = scaler.transform(user_data)
    pred_cgpa = model.predict(user_data_scaled)[0]
    pred_cgpa = np.clip(pred_cgpa, 0, 10)
    
    # Display the result
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted CGPA", f"{pred_cgpa:.2f}")
    with col2:
        st.metric("Target CGPA", f"{target_cgpa:.2f}")
    
    diff = target_cgpa - pred_cgpa
    if diff <= 0:
        st.success(f"🎉 Great job! You are on track to exceed your target CGPA by {-diff:.2f} points.")
        st.balloons()
    else:
        st.warning(f"⚠️ You are falling short of your target by {diff:.2f} points.")
        st.info("💡 **Tips to improve:** Try increasing your daily study hours, attending more classes, or finishing more assignments!")

# ================= TAB 2: Progress Analysis =================
with tab2:
    st.header("Your CGPA Trend")
    st.markdown("Enter your past semesters' CGPA below to visualize your progress:")
    
    # Initialize history tabular data
    if 'history' not in st.session_state:
        st.session_state['history'] = pd.DataFrame({
            "Semester": ["Sem 1", "Sem 2", "Sem 3", "Sem 4"],
            "CGPA": [7.0, 7.2, 7.5, 7.8]
        })
    
    edited_history = st.data_editor(st.session_state['history'], num_rows="dynamic", key="history_editor")
    st.session_state['history'] = edited_history
    
    if not edited_history.empty:
        # Plotting the timeline with plotly express
        fig = px.line(edited_history, x="Semester", y="CGPA", markers=True, title="Semester-wise CGPA Trend")
        fig.update_layout(yaxis=dict(range=[0, 10]))
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: Study Timetable =================
with tab3:
    st.header("Personalized Study Timetable")
    st.markdown("Add your subjects to generate a study schedule prioritized by credits and difficulty.")
    
    # Initialize subject data
    if 'subjects' not in st.session_state:
        st.session_state['subjects'] = pd.DataFrame({
            "Subject": ["Math", "Physics", "Computer Science"],
            "Credits": [4, 3, 4],
            "Difficulty (1-10)": [8, 7, 5],
            "Is Weak Area?": [True, False, False]
        })
    
    edited_subjects = st.data_editor(st.session_state['subjects'], num_rows="dynamic", key="subjects_editor")
    st.session_state['subjects'] = edited_subjects
    
    if st.button("Generate Timetable"):
        if edited_subjects.empty:
            st.error("Please add at least one subject!")
        else:
            # We calculate priority scores for subjects dynamically
            df_sched = edited_subjects.copy()
            df_sched['Priority_Score'] = (df_sched['Credits'].fillna(0) * 1.5) + (df_sched['Difficulty (1-10)'].fillna(0) * 2.0) + (df_sched['Is Weak Area?'].fillna(False).astype(int) * 5)
            df_sched = df_sched.sort_values(by='Priority_Score', ascending=False).reset_index(drop=True)
            
            total_priority = df_sched['Priority_Score'].sum()
            if total_priority == 0:
                total_priority = 1
                
            # Distribute the daily study hours (provided in sidebar) based on priority
            df_sched['Daily Suggested Hours'] = (df_sched['Priority_Score'] / total_priority) * study_horas_in
            df_sched['Daily Suggested Hours'] = df_sched['Daily Suggested Hours'].round(1)
            
            st.subheader("Your Study Plan")
            st.dataframe(df_sched[['Subject', 'Daily Suggested Hours', 'Priority_Score']], use_container_width=True)
            
            fig_bar = px.bar(df_sched, x='Subject', y='Daily Suggested Hours', title="Daily Hours Allocation", color='Subject')
            st.plotly_chart(fig_bar, use_container_width=True)