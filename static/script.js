if (!sessionStorage.getItem('active_session')) {
    fetch('/api/reset-session', { method: 'POST' })
        .then(() => {
            sessionStorage.setItem('active_session', 'true');
            window.location.reload();
        });
}
let simInterval = null;
let mainChart = null;

// Initialize when page loads
window.onload = () => {
    initChart();
    refreshStats();
    // Auto-refresh stats every 3 seconds to catch CSV uploads
    setInterval(refreshStats, 3000);
};

function initChart() {
    const ctx = document.getElementById('mainChart');
    if (!ctx) return;
    mainChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['STABLE', 'ELEVATED', 'CRITICAL'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: ['#22c55e', '#f97316', '#ef4444'],
                borderWidth: 0,
                hoverOffset: 20
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            cutout: '80%'
        }
    });
}

async function refreshStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        // Update the 4 top cards
        document.getElementById('stat-total').innerText = data.total.toLocaleString();
        document.getElementById('stat-fraud').innerText = data.fraud.toLocaleString();
        document.getElementById('stat-rate').innerText = data.rate + "%";
        document.getElementById('stat-risk').innerText = data.avg_risk.toFixed(2);

        // Update Chart
        if (mainChart) {
            mainChart.data.datasets[0].data = [data.dist.low, data.dist.med, data.dist.high];
            mainChart.update();
        }
    } catch (err) {
        console.error("Dashboard Sync Error:", err);
    }
}

async function toggleSimulation() {
    const btn = document.getElementById('simBtn');
    const feed = document.getElementById('liveFeed'); // Select the feed container

    if (simInterval) {
        clearInterval(simInterval);
        simInterval = null;
        btn.innerText = "START SIMULATION";
        btn.classList.replace('bg-red-600', 'bg-blue-600');
    } else {
        // Clear the "System idle" message when starting
        if (feed.querySelector('p')) feed.innerHTML = ''; 

        btn.innerText = "STOP SIMULATION";
        btn.classList.replace('bg-blue-600', 'bg-red-600');

        simInterval = setInterval(async () => {
            const simData = {
                amt: (Math.random() * 1500).toFixed(2),
                distance: (Math.random() * 300).toFixed(1), 
                age: 30, hour: new Date().getHours(), ampm: 'PM', 
                category: 'grocery_pos', gender: 'M', state: 'NY', job: 'IT',
                lat: 40.7, long: -74.0, merch_lat: 40.71, merch_long: -74.01
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(simData)
                });
                const result = await res.json();

                // --- NEW UI LOGIC: ADDING THE ROW TO THE LIVE FEED ---
                const entry = document.createElement('div');
                entry.className = "animate-slide-in p-3 glass border-l-2 " + 
                                 (result.prediction === 1 ? "border-red-500 bg-red-500/5" : "border-green-500 bg-green-500/5") + 
                                 " rounded-lg text-[10px] flex justify-between items-center";
                
                entry.innerHTML = `
                    <div>
                        <p class="font-black">$${result.amt} <span class="text-slate-500 font-normal">| ${result.category}</span></p>
                        <p class="${result.prediction === 1 ? "text-red-400" : "text-green-400"} font-bold">${result.risk_level}</p>
                    </div>
                    <div class="text-right">
                        <p class="text-slate-500 font-mono">${new Date().toLocaleTimeString()}</p>
                        <p class="font-bold">${(result.probability * 100).toFixed(1)}%</p>
                    </div>
                `;

                feed.prepend(entry); // Add newest prediction to the top

                // Keep only the last 10 entries so it doesn't lag the browser
                if (feed.children.length > 10) {
                    feed.removeChild(feed.lastChild);
                }

                refreshStats(); // Update the big cards at the top
            } catch (err) {
                console.error("Simulation Feed Error:", err);
            }
        }, 2000);
    }
}
