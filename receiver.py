from flask import Flask, jsonify, render_template_string, request
import time

app = Flask(__name__)

# In-memory report storage for demonstration,
REPORTS = []

SENDER_IP = "10.37.107.203"  # <-- replace as needed

@app.route('/report', methods=['POST'])
def receive_report():
    data = request.json
    data['received_at'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    REPORTS.append(data)
    if len(REPORTS) > 100:
        REPORTS.pop(0)
    return jsonify({"status": "received"})

@app.route('/reports')
def get_reports():
    return jsonify({"reports": REPORTS[::-1]})

@app.route('/')
def dashboard():
    return render_template_string('''
    <html>
    <head>
        <title>Real-Time Posture & Fatigue Reports</title>
        <style>
            body { font-family: Arial, sans-serif; }
            table { border-collapse: collapse; width: 100%; background: #f8faff; }
            th, td { border: 1px solid #ccc; padding: 8px 10px; text-align: center; }
            th { background: #e9ecef; }
            tr:nth-child(even) { background: #f3f6fa; }
            .small { font-size: 0.95em; color: #888; }
        </style>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
        function safeFixed(val, digits) {
            if (typeof val === "number" && !isNaN(val)) return val.toFixed(digits);
            return "";
        }
        function renderReports(reports) {
            if (!reports.length) {
                document.getElementById('report-section').innerHTML = "<em>No reports yet.</em>";
                return;
            }
            let html = `<table><thead><tr>
                <th>User</th>
                <th>Session Start</th>
                <th>Session End</th>
                <th>Bad<br>Posture<br>Time (s)</th>
                <th>Bad<br>Posture<br>Events</th>
                <th>Inattentive<br>Time (s)</th>
                <th>Yawn<br>Count</th>
                <th>Total<br>Frames</th>
                <th>Total<br>Paused (s)</th>
                </tr></thead><tbody>`;
            for (let r of reports) {
                html += `<tr>
                    <td>${r.user ? r.user : ''}</td>
                    <td>${r.session_start ? r.session_start : ''}</td>
                    <td>${r.session_end ? r.session_end : ''}</td>
                    <td>${safeFixed(r.bad_posture_time, 1)}</td>
                    <td>${typeof r.bad_posture_events === "number" ? r.bad_posture_events : (r.bad_posture_events || '')}</td>
                    <td>${safeFixed(r.inattentive_time, 1)}</td>
                    <td>${typeof r.yawn_count === "number" ? r.yawn_count : (r.yawn_count || '')}</td>
                    <td>${typeof r.frames === "number" ? r.frames : (r.frames || '')}</td>
                    <td>${safeFixed(r.total_paused_time, 1)}</td>
                    </tr>`;
            }
            html += '</tbody></table>';
            document.getElementById('report-section').innerHTML = html;
        }
        function fetchReports() {
            $.get('/reports', function(data) {
                renderReports(data.reports);
            });
        }
        setInterval(fetchReports, 5000);
        window.onload = fetchReports;
        </script>
    </head>
    <body>
        <h1>Real-Time Posture & Eye Fatigue Reports</h1>
        <div style="margin-bottom:2em">
            <b>Live Camera Feed from Sender:</b><br>
            <img src="http://{{sender_ip}}:5001/video_feed" width="400" height="300" style="border:1px solid #bbb; border-radius:8px" alt="Live Camera Feed">
        </div>
        <div id="report-section">Loading...</div>
    </body>
    </html>
    ''', sender_ip=SENDER_IP)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)