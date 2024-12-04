document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('storm');
    const loadingMessage = document.getElementById('loadingMessage');

    // Form submission handling
    form.addEventListener('submit', (event) => {
        event.preventDefault();
        loadingMessage.style.display = 'block';
        loadingMessage.scrollIntoView({ behavior: 'smooth' });

        const topic = form.topic.value;

        // Send form data to the server
        const formData = new FormData(form);
        fetch('/loading', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) throw new Error(`Erreur HTTP : ${response.status}`);
            return response.text();
        })
        .then(data => {
            document.body.innerHTML = data; // Update page content
            startCheckingTask(topic);
        })
        .catch(error => console.error('Erreur lors de l\'envoi du formulaire:', error));
    });
});

// Start periodic task status checks
function startCheckingTask(topic) {
    setInterval(() => {
        checkTaskStatus(topic);
    }, 1000);
}

// Function to check the task status
function checkTaskStatus(topic) {
    const bloc_topic = document.getElementById('topic_truncated')
    const topic_truncated = bloc_topic.getAttribute("data-topic")
    const encodedTopic = encodeURIComponent(topic);
    fetch(`/check_status/${encodedTopic}`)
        .then(response => {
            if (!response.ok) throw new Error(`Erreur HTTP : ${response.status}`);
            return response.json();
        })
        .then(data => {
            document.querySelector('#status').innerText = `Statut de la tâche : ${data.status}`;
            console.log(data.status)
            if (data.status === 'Redirection vers la page de résultats...') {
                const encodedTopicTruncated = encodeURIComponent(topic_truncated);
                setTimeout(() => {
                window.location.href = `/result?filename=${encodedTopicTruncated}`;
            }, 3000);
        }
        })
        .catch(error => console.error('Erreur lors de la vérification du statut:', error));
}