### **Map 1: Healthcare Infrastructure Deficit Map**

**The Goal:** Prove that physical distance to clinics is a geographic failure, not just a poverty issue.

```python
formula = "distance_problem_binary ~ wealth_index + barrier_transport + residence_type + education_level + barrier_money"

```

* **The Outcome:** `distance_problem_binary` (Does the respondent face a distance barrier?)
* **The Target (Treatment):** `wealth_index` (Continuous 1-5)
* **The Causal Logic:** By explicitly controlling for `barrier_transport` (can they afford the bus?) and `barrier_money` (can they afford the clinic?), we stripped away the financial aspects of travel. If `wealth_index` still had no effect (a coefficient near 0), it mathematically proved the clinics physically do not exist nearby.

---

### **Map 2: Healthcare Agency in relation to Financial Agency**

**The Goal:** Determine if giving a woman control over her own money translates into bodily autonomy.

```python
formula = "health_agency_binary ~ has_financial_agency + literacy_binary + education_level + wealth_index + age + internet_binary + tv_binary + C(residence_type) + C(caste) + C(religion)"

```

* **The Outcome:** `health_agency_binary` (Does she have a say in her healthcare?)
* **The Target (Treatment):** `has_financial_agency` (Does she control her own money?)
* **The Causal Logic:** Controlled for the wealth, education, literacy and modern media exposure.

---

### **Map 3: Impact of Television vs. Formal Education on Healthcare Autonomy**

**The Goal:** Compare the intensity of media consumption against formal schooling to see which drives autonomy faster.

```python
formula = "health_agency_binary ~ tv_z + edu_z + wealth_index + age + C(residence_type)"

```

* **The Outcome:** `health_agency_binary` (Does she have a say in her healthcare?)
* **The Targets (Treatments):** `tv_z` (Standardized TV frequency) and `edu_z` (Standardized Education level).
* **The Causal Logic:** Because TV frequency is a 0-3 scale and Education is a 0-5 scale, we couldn't compare them directly. Before running this formula, converted both into **Z-scores** (`tv_z` and `edu_z`). This allowed us to calculate exactly how much autonomy is gained per *1 Standard Deviation increase* in either medium, creating a perfectly fair fight.

---

### **Map 4: Insurance Efficiency Evaluation**

**The Goal:** Find out if possessing government health insurance actually protects people from out-of-pocket financial barriers.

```python
formula = "barrier_money_bin ~ govt_insurance + caste_reserved + wealth_index + education_level + age + C(residence_type)"

```

* **The Outcome:** `barrier_money_bin` (Is money a barrier to seeking care?)
* **The Target (Treatment):** `govt_insurance` (Does the respondent have a government plan?)
* **The Causal Logic:** Included `caste_reserved` and `wealth_index` as critical confounders because structural bias dictates both who gets targeted for insurance drives, and who is most vulnerable to clinic up-charging.
