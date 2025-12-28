const systemMode = document.getElementById("systemMode");
const enrollmentPanel = document.getElementById("enrollmentPanel");
const recognitionInfo = document.getElementById("recognitionInfo");
const facesInfo = document.getElementById("facesInfo");
const attendanceBody = document.getElementById("attendanceBody");
const confirmBtn = document.getElementById("confirmEnrollment");

const poses = [
    "Look straight",
    "Turn left",
    "Turn right",
    "Look up",
    "Look down",
    "Tilt head left",
    "Tilt head right",
    "Smile",
    "Neutral face",
    "Move slightly back"
];

// Start camera only when dashboard loads
window.addEventListener("load", () => {
    document.getElementById("videoStream").src = "/video_feed";
});

// document.addEventListener("DOMContentLoaded", () => {
//     updateModeUI("recognition");
// });

fetch("/system/mode/recognition", { method: "POST" });

function exitSystem() {
    fetch("/camera/stop", { method: "POST" })
        .finally(() => {
            document.getElementById("videoStream").src = "";
            window.location.href = "/";
        });
}

function updateModeUI(mode) {
    const enrollPanel = document.getElementById("enrollment-panel");

    if (mode === "recognition") {
        enrollPanel.style.display = "none";
    } else if (mode === "enrollment") {
        enrollPanel.style.display = "block";
    }
}

// Collapsible sidebar (use existing HTML button)
const sidebar = document.getElementById("sidebar");
const toggleBtn = document.getElementById("toggleSidebar");

if (confirmBtn) confirmBtn.disabled = true;

/* Sidebar toggle */
toggleBtn.onclick = () => {
    sidebar.classList.toggle("collapsed");
    toggleBtn.textContent = sidebar.classList.contains("collapsed") ? "❯❯" : "❮❮";
};




// System mode change
systemMode.addEventListener("change", async () => {
    document.getElementById("videoStream").src = "";

    hideOverlay();
    enrollmentActive = false;

    updateModeUI(systemMode.value);

    await fetch(`/system/mode/${systemMode.value}`, { method: "POST" });
    document.getElementById("videoStream").src = "/video_feed";
});

/* Overlay helper */
function showOverlay(msg, color="#00ff99") {
    overlayMessage.innerText = msg;
    overlayMessage.style.color = color;
    overlayMessage.style.fontSize = "40px";
    overlayMessage.style.fontWeight = "600";
    overlayMessage.style.display = "block";
}

function hideOverlay() {
    overlayMessage.style.display = "none";
}

// Fetch live recognition info every second
async function fetchRecognition() {
    try {
        const res = await fetch("/recognition/live");
        const data = await res.json();

        if (data.faces && data.faces.length > 0) {
            facesInfo.innerHTML = data.faces.map(f =>
                `<strong>${f.display_name}</strong> | ${f.role} | ${f.access_status} | d=${f.distance !== null ? f.distance.toFixed(2) : "--"}`
            ).join("<br>");
        } else {
            facesInfo.innerHTML = "No faces recognized";
        }

        if (data.attendance && data.attendance.length > 0) {
            attendanceBody.innerHTML = data.attendance.map(a =>
                `<tr>
                    <td>${a.timestamp}</td>
                    <td>${a.person_id}</td>
                    <td>${a.status}</td>
                    <td>${a.source}</td>
                </tr>`
            ).join("");
        }
    } catch (err) {
        console.error(err);
    }
    setTimeout(fetchRecognition, 1000);
}

fetchRecognition();

let enrollmentActive = false;

async function captureLoop() {
    if (!enrollmentActive) return;

    try {
        const res = await fetch("/enroll/capture", { method: "POST" });
        const data = await res.json();

        if (data.status === "duplicate") {
            showOverlay("Person already exists. Restarting enrollment", "#ff4444");
            confirmBtn.disabled = true;
            setTimeout(captureLoop, 1500);
            return;
        }

        if (data.status === "ok") {
            showOverlay(`${poses[data.count - 1]} (${data.count}/10) | Quality ${data.quality}`);

            confirmBtn.disabled = !data.done;

            if (!data.done) {
                setTimeout(captureLoop, 700);
            } else {
                showOverlay("Capture complete ✔ Click Confirm");
            }
        } else {
            document.getElementById("enrollStatus").innerText = data.message || "Error during capture";
            setTimeout(captureLoop, 1000);
        }
    } catch (err) {
        console.error(err);
        setTimeout(captureLoop, 1500);
    }
}

// Enrollment buttons
document.getElementById("startEnrollment").addEventListener("click", async () => {
    // FORCE mode switch
    systemMode.value = "enrollment";
    systemMode.dispatchEvent(new Event("change"));

    await fetch("/enroll/start", { method: "POST" });

    enrollmentActive = true;
    showOverlay("Enrollment started. Look at camera.");

    captureLoop();
});

document.getElementById("confirmEnrollment").addEventListener("click", async () => {
    enrollmentActive = false;

    const payload = {
        display_name: document.getElementById("displayName").value,
        role: document.getElementById("role").value,
        department: document.getElementById("department").value,
        access_status: document.getElementById("accessStatus").value
    };

    const res = await fetch("/enroll/confirm", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
    });

    const data = await res.json();
    document.getElementById("enrollStatus").innerText = "Enrollment successful ✔";

    systemMode.value = "recognition";
    systemMode.dispatchEvent(new Event("change"));
});

// Force correct UI on page load
updateModeUI(systemMode.value);

document.getElementById("exitSystemBtn").addEventListener("click", () => {
    exitSystem();
});

/* End of script.js */