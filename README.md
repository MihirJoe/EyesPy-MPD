# TO DO #
[4/7 Demo video (30 sec)](https://youtu.be/FekopEXXwtw)

# Synchronization
- Sync red dot on plot with image display ✅
- Keep red dot static while secondary gray dot appears for mouse hover
- Images should update on mouse hover
- Measurement graphs automatically update with hover ✅
- Ability to click/"jump" to point on scrub bar


# Plot Specific
- Verify image timestamp as ratio of (eye image pairs):(CSV max timestamp) ✅
- Superimposed dropdown for VPF/MRD1/??? data
- Same y-axis range between VPF/MRD1/??? data
- Ability to view (1) separate OD/OS and (2) single graph
- Add tooltip for coordinate display on mouse hover
- Replace all instances of "VPH" with "VPF"
- Rename plot titles to "... vs Time"
- Add grid to plot

# UI Improvements
- Allow user to filter displayed measurements
- Bigger "Previous Pair and Next Pair" buttons
- Append "image pair" to fractional index (4/35 "image pair")
- Reposition fractional index below scrub bar
- Rename table titles to "OD (Right) Eye, OS (Left) Eye"
- Raise table titles vertically (currently overlapping)
- Used densely dotted blue (#0072B2) and a solid orange (#E69F00)  
  for colorblind visibility and clear distinction if printed
  
# Future Usability Studies
## 1. Clinician Workflow Integration
- **Purpose:**  
  Assess how efficiently and intuitively clinicians can navigate the GUI during common clinical review tasks.

- **Procedure:**  
  Participants will perform typical workflows (e.g., identifying image pairs with specific measurement anomalies) using the GUI while thinking aloud. Their sessions will be timed and observed to identify pain points.

- **Evaluation Metrics:**  
  - Task completion time  
  - Number and type of user errors  
  - NASA-TLX survey scores (mental load, frustration, ease of use)  
  - Observational notes (confusion points, repeated steps)

---

## 2. Graph Interaction Study
- **Purpose:**  
  Evaluate how effectively users interact with dynamic elements like the scrub bar, dual-dot tracking system, and hover tooltips.

- **Procedure:**  
  Users will scrub through time-series graphs, attempt to match timestamp data with image pairs, and use the tooltip to extract coordinates. Screen recordings and mouse tracking may be used.

- **Evaluation Metrics:**  
  - Accuracy of coordinate interpretation  
  - Number of successful jumps to intended timepoints  
  - Time to locate specific events on the graph  
  - Tooltip recall score (prompt-based)

---

## 3. Measurement Readability Study
- **Purpose:**  
  Determine if the UI layout (including labels, graph titles, and dropdowns) allows for easy and fast interpretation of measurement values.

- **Procedure:**  
  Present users with multiple versions of the GUI (varying font sizes, spacing, label formats) and ask them to complete specific reading or identification tasks.

- **Evaluation Metrics:**  
  - Time to identify specific metrics (e.g., VPF at timestamp X)  
  - User preference rankings  
  - Misinterpretation rate of measurement labels  
  - Open-ended feedback on clarity and label comprehension

---

## 4. Accessibility & Colorblind Testing
- **Purpose:**  
  Ensure all users, including those with color vision deficiencies, can accurately interpret graph elements and measurement data.

- **Procedure:**  
  Participants will be shown visualizations in different color schemes or through simulated filters for protanopia, deuteranopia, etc. They'll complete interpretation tasks (e.g., identifying which eye a line represents).

- **Evaluation Metrics:**  
  - Correctness of visual interpretation  
  - Confusion rates between colors  
  - Satisfaction with color palette (via Likert scale)  
  - Recommendations for improved accessibility

---

## 5. Comprehensive Satisfaction Survey
- **Purpose:**  
  Collect broad feedback on the usability, usefulness, and user satisfaction after using the GUI.

- **Procedure:**  
  Following task-based testing, users will complete a structured feedback survey (e.g., SUS) and provide optional open-ended comments about the overall experience.

- **Evaluation Metrics:**  
  - System Usability Scale (SUS) score  
  - Net Promoter Score (NPS)  
  - Thematic analysis of open-text responses  
  - Suggestions or common frustrations noted


