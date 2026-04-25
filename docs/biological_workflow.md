# CiliaTracks  
## Biological Workflow Protocol

---

## Table of Contents
1. [Overview](#1-overview)  
2. [Materials](#2-materials)  
3. [Microsphere Application](#3-microsphere-application)  
4. [Capturing Particle Movement](#4-capturing-particle-movement)  

---

## 1. Overview

CiliaTracks is a comprehensive Python package and dedicated website that enables clinicians and researchers to analyze cilia-driven particle motility and accurately predict PCD from high-speed video microscopy (HSVM) data.

To carry out analysis with CiliaTracks, HSVM data must first be collected through a standardized biological protocol. This workflow involves applying fluorescent microspheres to cultured airway liquid interface (ALI) cells and capturing their movement.

---

## 2. Materials

- 1% solids, 1.9 μm diameter green fluorescent polymer microspheres (Thermo Fisher Scientific, Cat. No. G0200)  
- 2% Nu-Serum media (Corning, Product No. 355100)  
- ALI-differentiated cultured airway epithelial cells in Transwells (6.5 mm or 12 mm diameter; Corning, Product Nos. 3472 & 3462)  
- Widefield inverted fluorescence microscope capable of capturing videos at 100 fps  

---

## 3. Microsphere Application

1. Dilute 1% solids, 1.9 μm diameter fluorescent microspheres 1:1000 in warm 2% Nu-Serum  
   - Incubate at 37 °C for a minimum of 10 minutes  

2. Prepare cultured cells (differentiated on Transwell plates)  
   - Gently wash the apical chamber with warm 2% Nu-Serum, rotating lightly for 1 minute  
   - Aspirate media  

3. Apply microspheres  
   - Add diluted fluorescent microsphere solution slowly to the center of the insert  
   - 20 μL for 6.5 mm Transwells  
   - 60 μL for 12 mm Transwells  

4. Incubate cells for 10 minutes at 37 °C  

---

## 4. Capturing Particle Movement

1. Turn on the microscope and environmental chamber, setting temperature to 37 °C  

2. Place cells from incubator onto the microscope  

3. Initialize microscope settings:  
   - Fusion EPI viewing mode  
   - GFP fluorescence channel  
   - 10× magnification
  

<img width="306" height="168" alt="Picture1" src="https://github.com/user-attachments/assets/cf2dd71c-46a6-4d05-8f87-888128cff8ee" />


4. Identify regions with strong bead movement  
   - Optimal regions are typically away from the center of the insert  

<img width="452" height="391" alt="Screenshot 2026-04-24 at 10 45 18 PM" src="https://github.com/user-attachments/assets/7e1b66cc-2fe7-474b-b82e-352c288ea849" />


5. Capture video  
   - Ensure proper focus and contrast  
   - Record a **4.45-second video at 100 fps** using 10× magnification  

6. Record the μm/pixel resolution for the imaging setup  

<img width="508" height="28" alt="Picture3" src="https://github.com/user-attachments/assets/20c55261-1c80-4327-9845-357ffa8b4ad1" />

7. Save videos as `.avi` files (MJPEG compression recommended)  

8. Capture a minimum of **3 videos per insert**  

<img width="716" height="369" alt="Screenshot 2026-04-24 at 10 43 49 PM" src="https://github.com/user-attachments/assets/f94e3e24-a38e-4059-a825-a077c088a1e5" />

<img width="685" height="338" alt="Screenshot 2026-04-24 at 10 46 02 PM" src="https://github.com/user-attachments/assets/58d07286-7d66-40ec-895f-dae88459723b" />

9. After imaging, rinse inserts with 2% Nu-Serum three times  

10. Proceed with analysis using **CiliaTracks**  

---
