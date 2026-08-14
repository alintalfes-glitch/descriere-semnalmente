# 🔍 Analiză Facială – Fișă de Semnalmente

Aplicație web **100% client-side** pentru generarea automată a unei fișe descriptive de semnalmente faciale (portret vorbit) pe baza a două fotografii: una din față și una din profil.

Folosește **MediaPipe Face Mesh** pentru extragerea a 468+ landmark-uri faciale și clasifică automat 12 categorii: frunte, nas, ochi, gură, bărbie, tipul feței, păr, sprâncene, barbă, mustață, urechi și semne particulare.

Rezultatele sunt **editabile manual** și pot fi salvate local în `localStorage` sau exportate ca fișier JSON.

---

## ✨ Caracteristici

- 🖼️ **Procesare locală** – imaginile NU părăsesc browserul; pixelii sunt eliminați din memorie după analiză.
- 🧠 **Detecție facială robustă** – MediaPipe Face Landmarker (GPU/CPU), 468+ puncte.
- 📋 **12 categorii de semnalmente** – inclusiv urechi (adăugate recent).
- ✏️ **Editare manuală** – fiecare rezultat poate fi corectat prin dropdown sau checkbox.
- 💾 **Persistență locală** – salvarea fișelor în `localStorage` sub chei unice.
- 📥 **Export JSON** – descărcare fișier `.json` cu fișa completă.
- 🔒 **Confidențialitate totală** – niciun server, niciun stocare de imagini.

---

## 📁 Structura fișierelor
