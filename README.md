## 🎶 June Code Pudding: Spotify Dashboard
This project delivers an interactive Plotly Dash web app that explores music trends using Spotify’s audio features and metadata. Built for both data enthusiasts and casual listeners, the dashboard enables users to discover how musical attributes like energy, danceability, valence, and popularity evolve across genres and time.

Bonus features include song clustering using K-Means, track similarity discovery, and audio previews directly from Spotify.

## 🧠 Key Insights & Storytelling
🎤 Pop and hip-hop dominate recent years, with strong upward trends in energy and danceability.

📈 Popularity is tightly linked to high energy and danceability across most genres.

🧭 Genre-specific musical “profiles” show clear distinctions in style and composition.

🔍 K-Means clustering groups similar tracks, helping users find songs that match a selected mood or style—even across artists.

Whether you're analyzing what makes a song go viral or just exploring your favorite genre over time, this dashboard helps you see the data behind the music.

## 🚀 Features
🎛️ Interactive Dashboard: Real-time filtering by year, genre, artist, and individual tracks.

📊 Visualizations Across Genres and Time: Including line plots, box plots, bar charts, and polar profiles.

🎧 Spotify Previews: Play short previews of songs from your selected genre.

🤖 Clustering & Similarity: K-Means with PCA lets you explore song clusters and similar tracks.

🧹 Data Cleaning Notebook: Reproducible transformations for consistency and clarity.

## 🗃️ Dataset
Sourced from:
Kaggle Spotify Top 2000s Mega Dataset
Includes top songs from the 2000s to early 2020s with audio feature metadata from Spotify.

## 📁 Project Structure
```
spotify-dashboard/
├── data/
│   └── spotify_dataset.csv
├── notebooks/
│   └── data_cleaning.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## 🏆 Bonus Features
✅ K-Means clustering with PCA-based visualization

✅ Cosine similarity for “similar song” discovery

✅ Embedded Spotify previews for user-selected tracks

✅ Storytelling-driven UI with dynamic chart updates

## 🌐 Live Demo 
Check out the deployed version of our project here: [Interactive Music Trends Dashboard](https://june-code-jam.onrender.com/).

Explore the interactive visualizations and insights generated from the data we’ve analyzed!

## 💡 Future Improvements
Add user authentication and playlists

Connect to Spotify API for live updates or more recent songs

Implement a recommendation engine based on user preferences

## 🤝 Credits
This project was created as part of the TripleTen Data Science program. Special thanks to:
  - TripleTen instructors and peers for ongoing support and feedback
  - Kaggle for the dataset

## 🛡️ License
This project is licensed under the MIT License.
