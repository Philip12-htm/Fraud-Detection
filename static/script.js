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
    if (simInterval) {
        clearInterval(simInterval);
        simInterval = null;
        btn.innerText = "START SIMULATION";
        btn.classList.replace('bg-red-600', 'bg-blue-600');
    } else {
        btn.innerText = "STOP SIMULATION";
        btn.classList.replace('bg-blue-600', 'bg-red-600');
        simInterval = setInterval(async () => {
            const simData = {
                amt: (Math.random() * 1500).toFixed(2),
                distance: 50, age: 30, hour: 12, ampm: 'PM', 
                category: 'grocery_pos', gender: 'M', state: 'NY', job: 'IT',
                lat: 40.7, long: -74.0, merch_lat: 40.71, merch_long: -74.01
            };
            await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(simData)
            });
            refreshStats();
        }, 2000);
    }
}