document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const form = document.getElementById('uploadForm');

    if (fileInput && imagePreview) {
        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.innerHTML = `
                        <h2>Image Preview:</h2>
                        <img src="${e.target.result}" alt="Selected Leaf">
                    `;
                };
                reader.readAsDataURL(file);
            } else {
                imagePreview.innerHTML = '';
            }
        });
    }

    if (form) {
        form.addEventListener('submit', () => {
            imagePreview.innerHTML += '<p>Processing...</p>';
        });
    }
});
