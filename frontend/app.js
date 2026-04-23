document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('searchForm');
    const input = document.getElementById('promptInput');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btnLoader');
    const searchBtn = document.getElementById('searchBtn');
    
    const resultSection = document.getElementById('resultSection');
    const geminiReply = document.getElementById('geminiReply');
    const agentThinking = document.getElementById('agentThinking');
    const moviesGrid = document.getElementById('moviesGrid');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const prompt = input.value.trim();
        if (!prompt) return;

        // UI State: Loading
        btnText.style.display = 'none';
        btnLoader.style.display = 'block';
        searchBtn.disabled = true;
        
        resultSection.style.display = 'block';
        geminiReply.style.display = 'none';
        agentThinking.style.display = 'block';
        moviesGrid.innerHTML = ''; // clear previous

        try {
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, k: 5 })
            });

            if (!response.ok) {
                throw new Error('Hubo un error al procesar tu solicitud.');
            }

            const data = await response.json();
            
            // Format Gemini Reply (basic markdown bold parsing for better looks)
            let formattedReply = data.reply.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            formattedReply = formattedReply.replace(/\n/g, '<br>');
            
            geminiReply.innerHTML = formattedReply;

            // Render Movie Cards
            data.movies.forEach((movie, idx) => {
                const card = document.createElement('div');
                card.className = 'movie-card';
                card.style.animation = `float 0.5s ease forwards ${idx * 0.1}s`;
                card.style.opacity = '0';
                
                card.innerHTML = `
                    <div class="card-header">
                        <div>
                            <h3 class="movie-title">${movie.title}</h3>
                            <span class="movie-year">${movie.year}</span>
                        </div>
                        <div class="rating-badge">★ ${movie.rating}</div>
                    </div>
                    <div class="movie-meta">
                        <div><span class="meta-label">Géneros:</span> ${movie.genres || 'N/A'}</div>
                        <div style="margin-top: 0.5rem;"><span class="meta-label">Elenco:</span> ${movie.cast || 'N/A'}</div>
                    </div>
                    <div class="movie-overview">
                        ${movie.overview || 'Sin descripción.'}
                    </div>
                `;
                moviesGrid.appendChild(card);
            });

            // UI State: Success
            agentThinking.style.display = 'none';
            geminiReply.style.display = 'block';

        } catch (error) {
            console.error(error);
            agentThinking.style.display = 'none';
            geminiReply.style.display = 'block';
            geminiReply.innerHTML = `<span style="color: #ef4444;">Ocurrió un error consultando el servidor. Verifica que esté corriendo y el archivo .env esté configurado correctamente.</span>`;
        } finally {
            // UI State: Reset Button
            btnText.style.display = 'block';
            btnLoader.style.display = 'none';
            searchBtn.disabled = false;
        }
    });

    // Add keyframe dynamic injection for cascade animation on cards
    const style = document.createElement('style');
    style.textContent = `
        @keyframes float {
            0% { transform: translateY(20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
});
