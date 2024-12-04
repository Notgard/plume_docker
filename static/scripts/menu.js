document.addEventListener('DOMContentLoaded', () => {
    const radioButtons = {
        local: document.getElementById('local'),
        cloud: document.getElementById('cloud'),
        internet: document.getElementById('internet'),
    };
    const localSection = document.getElementById('localSection');
    const fileInput = document.getElementById('fileInput');
    const labelText = document.getElementById('labelText');
    const resultMessage = document.getElementById('resultMessage');

    // Function to toggle the local section visibility
    function toggleSections() {
        localSection.style.display = radioButtons.local.checked ? 'block' : 'none';
    }

    toggleSections();

    Object.values(radioButtons).forEach(radio => {
        radio.addEventListener('change', toggleSections);
    });

    fileInput.addEventListener('change', () => {
        const files = fileInput.files;
        labelText.textContent = files.length > 0 
            ? `${files.length} fichier(s) sélectionné(s)` 
            : 'Cliquez ou glissez vos dossiers ici';
        resultMessage.textContent = files.length > 0 
            ? `Vous avez ajouté ${files.length} fichier(s) à télécharger.` 
            : '';
        resultMessage.style.color = files.length > 0 ? 'green' : '';
    });
});

document.addEventListener('DOMContentLoaded', () => {
    const masquesCheckbox = document.getElementById("montrer");
    const advancedOptions = document.getElementById('advancedOptions');

    // Initial state setup
    advancedOptions.style.display = masquesCheckbox.checked ? 'block' : 'none';

    // Event listener for checkbox change
    masquesCheckbox.addEventListener('change', () => {
        advancedOptions.style.display = masquesCheckbox.checked ? 'block' : 'none';
    });
});

document.addEventListener('DOMContentLoaded', () => {
    const content = document.querySelector('.file_content');
    content.innerHTML = content.textContent; // Clear the content

    const returnButton = document.getElementById("retourPage");
    returnButton.addEventListener("click", () => {
        window.open("/", "_self"); // Navigate to home page
    });
});