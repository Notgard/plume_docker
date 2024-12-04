document.addEventListener('DOMContentLoaded', () => {
    const content = document.querySelector('.file_content');
    content.innerHTML = content.textContent; // Clear the content

    const returnButton = document.getElementById("retourPage");
    returnButton.addEventListener("click", () => {
        window.open("/", "_self"); // Navigate to home page
    });
});