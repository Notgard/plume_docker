document.addEventListener('DOMContentLoaded', () => {
    const newUserButton = document.getElementById("newUser");
    const newUserDiv = document.getElementById("newUserDiv");

    // Show new user section on button click
    newUserButton.addEventListener('click', () => {
        newUserDiv.style.visibility = 'visible';
    });
});