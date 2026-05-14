
from flask import Flask, render_template, request, redirect, session
import sqlite3
import pickle
import numpy as np
import smtplib
import pandas as pd
from email.mime.text import MIMEText
from sklearn.metrics import confusion_matrix

app = Flask(__name__)
app.secret_key = "secretkey"

MODEL_OPTIONS = ["RandomForest", "KNN"]
DEFAULT_MODEL = "RandomForest"


def get_active_model():
    selected_model = session.get("selected_model", DEFAULT_MODEL)
    model_paths = {
        "RandomForest": "model_rf.pkl",
        "KNN": "model_knn.pkl"
    }

    model_file = model_paths.get(selected_model, "model.pkl")

    try:
        loaded_model = pickle.load(open(model_file, "rb"))
        return loaded_model, selected_model
    except Exception as e:
        print(f"error loading model {model_file}: {e}")
        try:
            loaded_model = pickle.load(open("model.pkl", "rb"))
            session["selected_model"] = DEFAULT_MODEL
            return loaded_model, DEFAULT_MODEL
        except Exception as e2:
            raise RuntimeError(f"Failed to load any model: {e2}")


# ============ EMAIL CONFIGURATION ============
# IMPORTANT: Update these credentials with your own Gmail account
# Steps to enable:
# 1. Go to: https://myaccount.google.com/apppasswords
# 2. Select "Mail" and "Windows Computer" (or your device)
# 3. Copy the app password and paste below
EMAIL_CONFIG = {
    'sender_email': 'majorpsi404@gmail.com',  # Your Gmail address
    'app_password': 'hkayejgdbmfbuslk',     # 16-character app password from Google
    'enabled': True  # Set to True once credentials are configured
}

WARNING_SUBJECT = "⚠️ Official Academic Warning Notice"

WARNING_BODY = """
Dear Student,

This is to inform you that your current academic performance and attendance record have fallen below the required standards set by ABC College.

Our records indicate the following concerns:

• Low Academic Performance
• Low Attendance
• High Risk of Failure

This serves as an official warning. You are strongly advised to:

• Attend all upcoming classes regularly
• Improve your academic performance
• Meet faculty members for guidance
• Attend remedial classes if available

Failure to improve may result in further disciplinary action.

Wishing you success in your academic progress.

Sincerely,
Principal
ABC College
"""

FEE_REMINDER_SUBJECT = "⚠️ College Fee Payment Reminder"

FEE_REMINDER_BODY = """
Dear Student,

This is to inform you that your **college fees are still pending** as per our records.

📌 **Status:** Not Paid
📌 **Action Required:** Please clear your dues at the earliest to avoid academic restrictions.

Failure to pay the fees may result in:

* Restriction from examinations
* Withholding of results
* Suspension from classes

If you have already made the payment, please ignore this message or contact the accounts department.

Regards,
**ABC College**
Accounts Department
"""

def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  parent_email TEXT,
                  attendance REAL,
                  assignment REAL,
                  study_hours REAL,
                  mid_marks REAL,
                  prediction TEXT,
                  probability REAL,
                  risk TEXT)''')
    conn.commit()
    conn.close()

init_db()

def send_email(parent_email, risk, probability, student_data=None):
    """Send detailed performance alert email to parent"""
    if not EMAIL_CONFIG['enabled']:
        print(f"Email notifications disabled. Would send to: {parent_email}")
        return True
    
    try:
        sender_email = EMAIL_CONFIG['sender_email']
        app_password = EMAIL_CONFIG['app_password']
        
        subject = "⚠️ Student Performance Alert - Action Required"
        
        # Build detailed email body
        if student_data:
            body = f"""Dear Parent,

We are writing to inform you about your child's academic performance and learning progress.

════════════════════════════════════════════════════════════════
                    PERFORMANCE SUMMARY
════════════════════════════════════════════════════════════════

📊 RISK CLASSIFICATION: {risk}
📈 Probability of Passing: {probability}%

════════════════════════════════════════════════════════════════
                    DETAILED METRICS
════════════════════════════════════════════════════════════════

📅 Attendance Rate:        {student_data.get('attendance', 'N/A')}%
📋 Assignment Performance: {student_data.get('assignment', 'N/A')}/100
⏰ Study Hours per Week:   {student_data.get('study_hours', 'N/A')} hours
🎓 Mid-term Marks:        {student_data.get('mid_marks', 'N/A')}/100

════════════════════════════════════════════════════════════════
                    RECOMMENDATIONS
════════════════════════════════════════════════════════════════

Based on the analysis, we recommend:

✓ Regular communication with the class teacher
✓ Increase study time and focus on weak areas
✓ Improve attendance and participation in class
✓ Schedule a meeting with academic counselors
✓ Consider additional tutoring or support programs

════════════════════════════════════════════════════════════════

Please contact the school administration if you have any questions
or concerns regarding your child's performance.

Best Regards,
Student Analytics & Performance Monitoring System
School Administration

Discipline | Dedication | Excellence"""
        else:
            body = f"""Dear Parent,

We are writing to inform you about your child's academic performance.

⚠️ RISK CLASSIFICATION: {risk}
📈 Probability of Passing: {probability}%

Please contact the school administration for more details.

Best Regards,
Student Analytics System"""
        
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = parent_email
        
        # Send email via Gmail SMTP
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, parent_email, msg.as_string())
        server.quit()
        
        print(f"✓ Email successfully sent to: {parent_email}")
        return True
        
    except Exception as e:
        print(f"✗ Error sending email to {parent_email}: {str(e)}")
        return False


def send_warning_email(student):
    """Send personalized score email to student"""

    if not EMAIL_CONFIG['enabled']:
        print(f"Email disabled for {student['email']}")
        return True

    try:
        sender_email = EMAIL_CONFIG['sender_email']
        app_password = EMAIL_CONFIG['app_password']

        recipient_email = student['email']

        subject = "📊 Student Performance Report"

        body = f"""
Dear Student,

Here is your academic performance report.

━━━━━━━━━━━━━━━━━━━━━━
📚 PERFORMANCE DETAILS
━━━━━━━━━━━━━━━━━━━━━━

Attendance      : {student['attendance']}%
Assignment Avg  : {student['assignment']}
Study Hours     : {student['study_hours']}
Mid Marks       : {student['mid_marks']}

━━━━━━━━━━━━━━━━━━━━━━
📈 PREDICTION RESULT
━━━━━━━━━━━━━━━━━━━━━━

Probability Score : {student['probability']}%
Risk Level        : {student['risk']}

━━━━━━━━━━━━━━━━━━━━━━

Please work hard and improve your performance.

Best Regards,
ABC College
"""

        msg = MIMEText(body)

        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()

        server.login(sender_email, app_password)

        server.sendmail(
            sender_email,
            recipient_email,
            msg.as_string()
        )

        server.quit()

        print(f"✓ Email sent to {recipient_email}")

        return True

    except Exception as e:
        print("EMAIL ERROR:", e)
        return False


def send_fee_reminder_email(student):
    """Send fee reminder email to student."""
    if not EMAIL_CONFIG['enabled']:
        print(f"Email notifications disabled. Would send to: {student['email']}")
        return True

    try:
        sender_email = EMAIL_CONFIG['sender_email']
        app_password = EMAIL_CONFIG['app_password']
        recipient_email = student['email']

        msg = MIMEText(FEE_REMINDER_BODY)
        msg["Subject"] = FEE_REMINDER_SUBJECT
        msg["From"] = sender_email
        msg["To"] = recipient_email

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()

        print(f"✓ Fee reminder sent to {recipient_email}")
        return True

    except Exception as e:
        print(f"✗ Error sending fee reminder to {student['email']}: {e}")
        return False
    
 
def get_high_risk_recipients_from_csv():
    """Return a list of high-risk recipient records from the CSV file."""
    try:
        df = pd.read_csv("student_data.csv")
    except Exception as e:
        print(f"Error reading student_data.csv: {e}")
        return []

    email_columns = [col for col in df.columns if col.lower() in {"email", "parent_email", "parentemail", "student_email", "studentemail"}]
    if not email_columns:
        return []

    email_col = email_columns[0]
    recipients = []
    model, _ = get_active_model()

    for idx, row in df.iterrows():
        if pd.isna(row[email_col]):
            continue

        try:
            features = [[row["Attendance"], row["Assignment_Avg"], row["Study_Hours"], row["Mid_Marks"]]]
            probability = model.predict_proba(features)[0][1]
        except Exception as e:
            print(f"Error evaluating CSV row {idx + 1}: {e}")
            continue

        if probability < 0.50:
            recipients.append({
               "email": str(row[email_col]).strip(),
               "risk": "High Risk",
               "roll_number": idx + 1,
               "attendance": row["Attendance"],
               "assignment": row["Assignment_Avg"],
               "study_hours": row["Study_Hours"],
               "mid_marks": row["Mid_Marks"],
               "probability": round(probability * 100, 2)

            })

    return recipients

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin123":
            session["user"] = "admin"
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/student_login", methods=["POST"])
def student_login():
    roll_number = request.form.get("roll_number", "").strip()
    if not roll_number.isdigit():
        return render_template("login.html", error_student="Please enter a valid numeric Roll Number.")

    roll_number = int(roll_number)
    try:
        df = pd.read_csv("student_data.csv")
    except Exception as e:
        print(f"Error reading student_data.csv: {e}")
        return render_template("login.html", error_student="Unable to load student data.")

    if "Roll_Number" in df.columns:
        student_rows = df[df["Roll_Number"] == roll_number]
    else:
        if roll_number < 1 or roll_number > len(df):
            student_rows = df.iloc[0:0]
        else:
            student_rows = df.iloc[[roll_number - 1]]

    if student_rows.empty:
        return render_template("login.html", error_student="Roll Number not found. Please verify and try again.")

    row = student_rows.iloc[0]
    try:
        features = [[row["Attendance"], row["Assignment_Avg"], row["Study_Hours"], row["Mid_Marks"]]]
        model, active_model = get_active_model()
        prediction_val = model.predict(features)[0]
        probability_val = model.predict_proba(features)[0][1] * 100
    except Exception as e:
        print(f"Error predicting student data: {e}")
        return render_template("login.html", error_student="Unable to evaluate student record.")

    if probability_val >= 75:
        risk_label = "Low Risk"
    elif probability_val >= 50:
        risk_label = "Medium Risk"
    else:
        risk_label = "High Risk"

    try:
        actual_result = "PASS" if int(row.get("Final_Result", 0)) == 1 else "FAIL"
    except Exception:
        actual_result = "N/A"

    student_detail = {
        "roll_number": roll_number,
        "attendance": float(row["Attendance"]),
        "assignment": float(row["Assignment_Avg"]),
        "study_hours": float(row["Study_Hours"]),
        "mid_marks": float(row["Mid_Marks"]),
        "prediction": "PASS" if int(prediction_val) == 1 else "FAIL",
        "probability": round(probability_val, 2),
        "risk": risk_label,
        "final_result": actual_result
    }

    if "Name" in row.index:
        student_detail["name"] = row["Name"]

    study_score = min(100, round(student_detail["study_hours"] * 16.6667, 2))
    probability_component = min(100, round(probability_val, 2))
    ca3_total = round(
        student_detail["attendance"] +
        student_detail["assignment"] +
        student_detail["mid_marks"] +
        study_score +
        probability_component,
        2
    )

    warnings = []
    if student_detail["attendance"] < 60:
        warnings.append({
            "title": "Attendance Alert",
            "message": "Attendance is below 60%. Regular classroom presence is essential.",
            "current": f"{student_detail['attendance']}%",
            "target": "≥ 60%"
        })
    if student_detail["assignment"] < 60:
        warnings.append({
            "title": "Assignment Warning",
            "message": "Assignment scores are low. Improve task completion and accuracy.",
            "current": f"{student_detail['assignment']}/100",
            "target": "≥ 60/100"
        })
    if student_detail["study_hours"] < 2:
        warnings.append({
            "title": "Low Study Time",
            "message": "Weekly study time is low. Increase study hours for better exam readiness.",
            "current": f"{student_detail['study_hours']}h/week",
            "target": "≥ 2h/week"
        })
    if student_detail["mid_marks"] < 50:
        warnings.append({
            "title": "Mid-term Marks Warning",
            "message": "Mid-term marks are below expectation. Focus on weak subjects.",
            "current": f"{student_detail['mid_marks']}/100",
            "target": "≥ 50/100"
        })
    if probability_val < 50:
        warnings.append({
            "title": "Risk Alert",
            "message": "Overall predicted pass probability is low. Seek tutoring and revision support.",
            "current": f"{round(probability_val, 2)}%",
            "target": "≥ 50%"
        })

    student_detail["study_score"] = study_score
    student_detail["ca3_total"] = ca3_total
    student_detail["ca3_max"] = 500
    student_detail["warnings"] = warnings

    return render_template("student_view.html", student=student_detail)

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")

    selected_model = session.get("selected_model", DEFAULT_MODEL)
    return render_template("dashboard.html", selected_model=selected_model)


@app.route("/set_model", methods=["POST"])
def set_model():
    if "user" not in session:
        return redirect("/")

    model_choice = request.form.get("model_choice", DEFAULT_MODEL)
    if model_choice not in MODEL_OPTIONS:
        model_choice = DEFAULT_MODEL

    session["selected_model"] = model_choice
    return redirect("/dashboard")

@app.route("/add_student", methods=["GET","POST"])
def add_student():
    if "user" not in session:
        return redirect("/")
    
    if request.method == "POST":
        parent_email = request.form["parent_email"]
        attendance = float(request.form["attendance"])
        assignment = float(request.form["assignment"])
        study_hours = float(request.form["study_hours"])
        mid_marks = float(request.form["mid_marks"])

        model, active_model = get_active_model()

        features = [[attendance, assignment, study_hours, mid_marks]]
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        if probability >= 0.75:
            risk = "Low Risk"
        elif probability >= 0.50:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        result = "PASS" if prediction[0] == 1 else "FAIL"

        if risk == "High Risk":
            student_info = {
                'attendance': attendance,
                'assignment': assignment,
                'study_hours': study_hours,
                'mid_marks': mid_marks
            }
            send_email(parent_email, risk, round(probability*100,2), student_info)

        try:
            df = pd.read_csv("student_data.csv")
        except Exception as e:
            print(f"Error reading student_data.csv: {e}")
            df = pd.DataFrame(columns=["Attendance", "Assignment_Avg", "Study_Hours", "Mid_Marks", "Final_Result", "email", "Fees_Status"])

        if "Fees_Status" not in df.columns:
            df["Fees_Status"] = "Not Paid"
        if "email" not in df.columns:
            df["email"] = ""
        if "Final_Result" not in df.columns:
            df["Final_Result"] = 0

        new_student_row = {
            "Attendance": attendance,
            "Assignment_Avg": assignment,
            "Study_Hours": study_hours,
            "Mid_Marks": mid_marks,
            "Final_Result": 1 if result == "PASS" else 0,
            "email": parent_email,
            "Fees_Status": "Not Paid"
        }

        try:
            df = pd.concat([df, pd.DataFrame([new_student_row])], ignore_index=True)
            df.to_csv("student_data.csv", index=False)
        except Exception as e:
            print(f"Error writing student_data.csv: {e}")

        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO students (parent_email, attendance, assignment, study_hours, mid_marks, prediction, probability, risk) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (parent_email, attendance, assignment, study_hours, mid_marks, result, probability*100, risk))
        conn.commit()
        conn.close()

        return render_template("add_student.html", result=result, probability=round(probability*100,2), risk=risk)

    return render_template("add_student.html")

@app.route("/view_students")
def view_students():
    if "user" not in session:
        return redirect("/")
    
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM students")
    data = c.fetchall()
    conn.close()
    return render_template("view_students.html", data=data)

@app.route("/analytics")
def analytics():
    if "user" not in session:
        return redirect("/")
    
    try:
        # Read CSV file
        df = pd.read_csv("student_data.csv")
        model, active_model = get_active_model()
        predictions = []
        risk_counts = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
        pass_fail_counts = {"PASS": 0, "FAIL": 0}
        low_performers = []
        emails_info = []
        y_true = []
        y_pred = []
        
        for idx, row in df.iterrows():
            features = [[row['Attendance'], row['Assignment_Avg'], row['Study_Hours'], row['Mid_Marks']]]
            prediction = model.predict(features)
            probability = model.predict_proba(features)[0][1]
            
            true_result = row['Final_Result']  # Assuming 1 for PASS, 0 for FAIL
            y_true.append(true_result)
            y_pred.append(prediction[0])
            
            # Determine risk level
            if probability >= 0.75:
                risk = "Low Risk"
            elif probability >= 0.50:
                risk = "Medium Risk"
            else:
                risk = "High Risk"
            
            result = "PASS" if prediction[0] == 1 else "FAIL"
            
            # Count statistics
            risk_counts[risk] += 1
            pass_fail_counts[result] += 1
            
            # Collect low performers (High Risk students)
            if risk == "High Risk":
                student_record = {
                    'index': idx + 1,
                    'attendance': row['Attendance'],
                    'assignment': row['Assignment_Avg'],
                    'study_hours': row['Study_Hours'],
                    'mid_marks': row['Mid_Marks'],
                    'probability': round(probability * 100, 2),
                    'risk': risk
                }
                low_performers.append(student_record)
                emails_info.append(student_record)
            
            predictions.append({
                'attendance': row['Attendance'],
                'assignment': row['Assignment_Avg'],
                'study_hours': row['Study_Hours'],
                'mid_marks': row['Mid_Marks'],
                'prediction': result,
                'probability': round(probability * 100, 2),
                'risk': risk
            })
        
        # Sort low performers by probability (lowest first)
        low_performers.sort(key=lambda x: x['probability'])
        
        # Calculate statistics
        total_students = len(df)
        avg_attendance = df['Attendance'].mean()
        avg_assignment = df['Assignment_Avg'].mean()
        avg_study_hours = df['Study_Hours'].mean()
        avg_mid_marks = df['Mid_Marks'].mean()
        
        # Prepare data for charts
        risk_chart_data = {
            'labels': list(risk_counts.keys()),
            'data': list(risk_counts.values()),
            'backgroundColor': ['#10b981', '#f59e0b', '#ef4444']
        }
        
        pass_fail_chart_data = {
            'labels': list(pass_fail_counts.keys()),
            'data': list(pass_fail_counts.values()),
            'backgroundColor': ['#667eea', '#764ba2']
        }
        
        # Create attendance trend data (bins for attendance ranges)
        attendance_bins = {
            '0-20%': 0,
            '20-40%': 0,
            '40-60%': 0,
            '60-80%': 0,
            '80-100%': 0
        }
        
        for idx, row in df.iterrows():
            attendance = row['Attendance']
            if attendance <= 20:
                attendance_bins['0-20%'] += 1
            elif attendance <= 40:
                attendance_bins['20-40%'] += 1
            elif attendance <= 60:
                attendance_bins['40-60%'] += 1
            elif attendance <= 80:
                attendance_bins['60-80%'] += 1
            else:
                attendance_bins['80-100%'] += 1
        
        attendance_trend_data = {
            'labels': list(attendance_bins.keys()),
            'data': list(attendance_bins.values()),
            'borderColor': '#667eea',
            'backgroundColor': 'rgba(102, 126, 234, 0.1)',
            'tension': 0.4
        }
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # cm is [[TN, FP], [FN, TP]] for binary classification where 0 is FAIL, 1 is PASS
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate accuracy
        total_predictions = tn + fp + fn + tp
        accuracy = ((tp + tn) / total_predictions * 100) if total_predictions > 0 else 0
        
        confusion_matrix_data = {
            'labels': ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)'],
            'data': [int(tn), int(fp), int(fn), int(tp)],
            'backgroundColor': ['#10b981', '#ef4444', '#f59e0b', '#3b82f6']
        }
        
        return render_template("analytics.html", 
                             risk_chart=risk_chart_data,
                             pass_fail_chart=pass_fail_chart_data,
                             attendance_trend=attendance_trend_data,
                             confusion_matrix=confusion_matrix_data,
                             total_students=total_students,
                             low_risk=risk_counts["Low Risk"],
                             medium_risk=risk_counts["Medium Risk"],
                             high_risk=risk_counts["High Risk"],
                             pass_count=pass_fail_counts["PASS"],
                             fail_count=pass_fail_counts["FAIL"],
                             accuracy=round(accuracy, 1),
                             avg_attendance=round(avg_attendance, 2),
                             avg_assignment=round(avg_assignment, 2),
                             avg_study_hours=round(avg_study_hours, 2),
                             avg_mid_marks=round(avg_mid_marks, 2),
                             low_performers=low_performers[:20])  # Top 20 lowest performers
    except Exception as e:
        print(f"Error in analytics: {e}")
        return redirect("/dashboard")

@app.route("/send_alerts", methods=["GET", "POST"])
def send_alerts():

    if "user" not in session:
        return redirect("/")

    results = {
        'sent': 0,
        'failed': 0,
        'disabled':0,
        'messages': []
    }

    # Get all high-risk students from CSV
    high_risk_students = get_high_risk_recipients_from_csv()

    if not high_risk_students:
        results['messages'].append("No high-risk students found.")
        return render_template("email_status.html", results=results)

    # Send emails automatically
    for student in high_risk_students:

        recipient = student["email"]

        success = send_warning_email(student)

        if success:
            results['sent'] += 1
            results['messages'].append(f"✓ Email sent to {recipient}")
        else:
            results['failed'] += 1
            results['messages'].append(f"✗ Failed to send email to {recipient}")

    return render_template("email_status.html", results=results)

@app.route("/fees_info", methods=["GET", "POST"])
def fees_info():
    if "user" not in session:
        return redirect("/")

    try:
        df = pd.read_csv("student_data.csv")
    except Exception as e:
        print(f"Error reading student_data.csv: {e}")
        return redirect("/dashboard")

    if "Fees_Status" not in df.columns:
        df["Fees_Status"] = "Not Paid"

    if request.method == "POST":
        action = request.form.get("action", "save_status")
        if action == "save_status":
            for idx, row in df.iterrows():
                roll_key = f"status_{idx + 1}"
                status = request.form.get(roll_key, "Not Paid")
                df.at[idx, "Fees_Status"] = status
            try:
                df.to_csv("student_data.csv", index=False)
                message = "Fee status updated successfully."
            except Exception as e:
                print(f"Error writing student_data.csv: {e}")
                message = "Unable to update fee status."

            students = [
                {
                    'roll_number': idx + 1,
                    'email': row.get('email', ''),
                    'fees_status': str(df.at[idx, 'Fees_Status'])
                }
                for idx, row in df.iterrows()
            ]

            return render_template("fees_info.html", students=students, message=message)

        if action == "send_reminders":
            unpaid_students = []
            for idx, row in df.iterrows():
                status = str(row.get('Fees_Status', 'Not Paid')).strip().lower()
                email = str(row.get('email', '')).strip()
                if status != "paid" and email:
                    unpaid_students.append({
                        'roll_number': idx + 1,
                        'email': email
                    })

            results = {'sent': 0, 'failed': 0, 'disabled': 0, 'messages': []}
            if not unpaid_students:
                results['messages'].append("No unpaid student emails to send.")
                return render_template("email_status.html", results=results)

            for student in unpaid_students:
                success = send_fee_reminder_email(student)
                if success:
                    results['sent'] += 1
                    results['messages'].append(f"✓ Email sent to {student['email']}")
                else:
                    results['failed'] += 1
                    results['messages'].append(f"✗ Failed to send email to {student['email']}")

            return render_template("email_status.html", results=results)

    students = [
        {
            'roll_number': idx + 1,
            'email': row.get('email', ''),
            'fees_status': str(row.get('Fees_Status', 'Not Paid'))
        }
        for idx, row in df.iterrows()
    ]

    return render_template("fees_info.html", students=students)

@app.route("/search_students", methods=["GET", "POST"])
def search_students():
    """Search and filter students with personalized recommendations"""
    if "user" not in session:
        return redirect("/")

    results = []
    search_query = ""
    filter_type = "all"

    if request.method == "POST":
        search_query = request.form.get("search_query", "").strip()
        filter_type = request.form.get("filter_type", "all")

        df = pd.read_csv("student_data.csv")
        model, active_model = get_active_model()

        for idx, row in df.iterrows():
            student_id = idx + 1
            student_name = row.get('Name', f'Student {student_id}')
            attendance = float(row['Attendance'])
            assignment = float(row['Assignment_Avg'])
            study_hours = float(row['Study_Hours'])
            mid_marks = float(row['Mid_Marks'])
            final_result = "PASS" if int(row['Final_Result']) == 1 else "FAIL"

            features = [[attendance, assignment, study_hours, mid_marks]]
            prediction_val = model.predict(features)[0]
            probability_val = model.predict_proba(features)[0][1] * 100

            if probability_val >= 75:
                risk_label = "Low Risk"
            elif probability_val >= 50:
                risk_label = "Medium Risk"
            else:
                risk_label = "High Risk"

            matches_search = (
                search_query == "" or
                search_query.lower() in str(student_name).lower() or
                search_query.lower() in str(attendance).lower() or
                search_query.lower() in str(assignment).lower() or
                search_query.lower() in str(study_hours).lower() or
                search_query.lower() in str(mid_marks).lower() or
                search_query.lower() in final_result.lower() or
                search_query.lower() in risk_label.lower()
            )

            if not matches_search:
                continue

            if filter_type == "low_attendance" and attendance >= 60:
                continue
            elif filter_type == "high_attendance" and attendance < 80:
                continue
            elif filter_type == "low_study" and study_hours >= 2:
                continue
            elif filter_type == "low_marks" and mid_marks >= 50:
                continue
            elif filter_type == "high_marks" and mid_marks < 75:
                continue
            elif filter_type == "high_risk" and risk_label != "High Risk":
                continue

            recommendations = []
            if attendance < 60:
                recommendations.append({
                    'icon': '📅',
                    'type': 'Attendance',
                    'message': 'Improve class attendance.',
                    'current': f'{attendance}%',
                    'target': '≥ 60%'
                })
            if study_hours < 2:
                recommendations.append({
                    'icon': '⏰',
                    'type': 'Study Time',
                    'message': 'Increase weekly study hours.',
                    'current': f'{study_hours}h/week',
                    'target': '≥ 2h/week'
                })
            if mid_marks < 50:
                recommendations.append({
                    'icon': '📚',
                    'type': 'Mid Marks',
                    'message': 'Focus on weak subjects.',
                    'current': f'{mid_marks}/100',
                    'target': '≥ 50/100'
                })

            results.append({
                'id': student_id,
                'name': student_name,
                'attendance': attendance,
                'assignment': assignment,
                'study_hours': study_hours,
                'mid_marks': mid_marks,
                'final_result': final_result,
                'prediction': "PASS" if prediction_val == 1 else "FAIL",
                'probability': round(probability_val, 2),
                'risk': risk_label,
                'recommendations': recommendations
            })

    return render_template("search_students.html", 
                         results=results,
                         search_query=search_query,
                         filter_type=filter_type)

@app.route("/update_marks", methods=["GET", "POST"])
def update_marks():
    if "user" not in session:
        return redirect("/")

    message = ""
    error = False

    if request.method == "POST":
        roll_number = request.form.get("roll_number", "").strip()
        if not roll_number.isdigit():
            message = "Please enter a valid numeric Roll Number."
            error = True
        else:
            roll_number = int(roll_number)
            try:
                df = pd.read_csv("student_data.csv")
            except Exception as e:
                message = f"Error reading student_data.csv: {e}"
                error = True
                return render_template("update_marks.html", message=message, error=error)

            # Find student by roll number
            student_row = None
            student_idx = None
            if "Roll_Number" in df.columns:
                matching_rows = df[df["Roll_Number"] == roll_number]
                if not matching_rows.empty:
                    student_row = matching_rows.iloc[0]
                    student_idx = matching_rows.index[0]
            else:
                if roll_number >= 1 and roll_number <= len(df):
                    student_idx = roll_number - 1
                    student_row = df.iloc[student_idx]

            if student_row is None:
                message = "Roll Number not found. Please verify and try again."
                error = True
            else:
                # Ensure new columns exist
                new_columns = [
                    "Artificial_Intelligence", "Machine_Learning", "Python_Programming",
                    "Data_Warehousing_and_Data_Mining", "DBMS", "Total_CA3",
                    "Assessment_Rubric", "Strengths", "Areas_for_Improvement", "Suggested_Corrective_Measures"
                ]
                for col in new_columns:
                    if col not in df.columns:
                        df[col] = ""

                # Get marks
                ai = request.form.get("ai", "").strip()
                ml = request.form.get("ml", "").strip()
                python = request.form.get("python", "").strip()
                dw = request.form.get("dw", "").strip()
                dbms = request.form.get("dbms", "").strip()

                marks = [ai, ml, python, dw, dbms]
                if not all(m for m in marks):
                    message = "Please enter valid numeric marks for all subjects."
                    error = True
                else:
                    try:
                        ai_val = float(ai)
                        ml_val = float(ml)
                        python_val = float(python)
                        dw_val = float(dw)
                        dbms_val = float(dbms)
                    except ValueError:
                        message = "Please enter valid numeric marks for all subjects."
                        error = True
                    else:
                        total_ca3 = ai_val + ml_val + python_val + dw_val + dbms_val

                        rubric = request.form.get("rubric", "").strip()
                        strengths = request.form.get("strengths", "").strip()
                        areas = request.form.get("areas", "").strip()
                        measures = request.form.get("measures", "").strip()

                        # Update the row
                        df.at[student_idx, "Artificial_Intelligence"] = ai_val
                        df.at[student_idx, "Machine_Learning"] = ml_val
                        df.at[student_idx, "Python_Programming"] = python_val
                        df.at[student_idx, "Data_Warehousing_and_Data_Mining"] = dw_val
                        df.at[student_idx, "DBMS"] = dbms_val
                        df.at[student_idx, "Total_CA3"] = total_ca3
                        df.at[student_idx, "Assessment_Rubric"] = rubric
                        df.at[student_idx, "Strengths"] = strengths
                        df.at[student_idx, "Areas_for_Improvement"] = areas
                        df.at[student_idx, "Suggested_Corrective_Measures"] = measures

                        try:
                            df.to_csv("student_data.csv", index=False)
                            message = f"Marks updated successfully for Roll Number {roll_number}. Total CA3: {total_ca3}"
                            error = False
                        except Exception as e:
                            message = f"Error saving data: {e}"
                            error = True

    return render_template("update_marks.html", message=message, error=error)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
