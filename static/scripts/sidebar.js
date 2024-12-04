function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const content = document.querySelector('.content');

    if (sidebar.style.left === '0px') {
        sidebar.style.left = '-250px';  // Hide the sidebar
        content.style.marginLeft = '0';  // Reset content margin
    } else {
        sidebar.style.left = '0';  // Show the sidebar
        content.style.marginLeft = '250px';  // Adjust content margin
    }
}