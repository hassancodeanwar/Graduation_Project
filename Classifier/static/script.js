// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.querySelector('.upload-btn');
    const previewSection = document.querySelector('.preview-section');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.querySelector('.analyze-btn');
    const results = document.querySelector('.results');
    const loading = document.querySelector('.loading');
    const classification = document.getElementById('classification');
    const confidence = document.getElementById('confidence');

    // Handle file selection via button
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#dcdde1';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#dcdde1';
        handleFiles(e.dataTransfer.files);
    });

    // Handle file input change
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // Handle file processing
    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewSection.style.display = 'block';
                    results.style.display = 'none';
                };
                reader.readAsDataURL(file);
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    // Handle image analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!previewImage.src) {
            alert('Please upload an image first.');
            return;
        }

        // Show loading spinner
        loading.style.display = 'block';
        analyzeBtn.disabled = true;

        // Create form data
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                classification.textContent = data.class;
                confidence.textContent = data.confidence.toFixed(1);
                results.style.display = 'block';
            } else {
                alert(data.error || 'An error occurred during analysis.');
            }
        } catch (error) {
            alert('An error occurred during analysis.');
            console.error('Error:', error);
        } finally {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });
});