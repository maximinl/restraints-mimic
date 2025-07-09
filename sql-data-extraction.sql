WITH first_icu_stay AS (
  -- Selects the first ICU stay per patient, including intime, outtime, hadm_id
  SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn
  FROM `physionet-data.mimiciv_icu.icustays`
),

-- Captures exact time of restraint documentation AND the stay's intime
restraint_charttimes AS (
  SELECT DISTINCT
    ce.subject_id,
    ce.stay_id,
    icu.intime, -- Added intime here
    ce.charttime AS restraint_time
  FROM `physionet-data.mimiciv_icu.chartevents` ce
  JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn=1
  WHERE ce.stay_id IS NOT NULL AND
  REGEXP_CONTAINS(LOWER(value), r'(?i)(restrain|tied|secured|immobili[sz]ed|4[- ]point|four[- ]point|extremities secured|limbs secured|wrist restraint|soft restraint|leather restraint)')
  AND ce.charttime BETWEEN icu.intime AND icu.outtime -- Ensures restraint is within stay
),

-- Find the time of the very FIRST restraint event per stay
first_restraint_time_cte AS (
  SELECT
    subject_id,
    stay_id,
    MIN(restraint_time) AS first_restraint_time
  FROM restraint_charttimes
  GROUP BY subject_id, stay_id
),

-- Counts distinct 24-HOUR PERIODS from admission time with restraint documentation
restraint_periods_count AS ( -- Renamed CTE for clarity
  SELECT
    subject_id,
    stay_id,
    -- Calculate the 0-indexed day number relative to admission
    COUNT(DISTINCT FLOOR(TIMESTAMP_DIFF(restraint_time, intime, HOUR) / 24)) AS restraint_24hr_periods_count
  FROM restraint_charttimes
  GROUP BY subject_id, stay_id
),

-- Clinical Status - First 24 Hours ---------------------------------------
ventilated_day1 AS (
  SELECT DISTINCT ce.subject_id, ce.stay_id
  FROM `physionet-data.mimiciv_icu.chartevents` ce
  JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR)
    AND REGEXP_CONTAINS(LOWER(ce.value), r'(?i)(intubat|mechanical ventilation|ventilated|endotracheal tube|ett|breathing tube|ventilator support|vent support)')
  UNION DISTINCT SELECT DISTINCT di.subject_id, ie.stay_id FROM `physionet-data.mimiciv_hosp.diagnoses_icd` di JOIN first_icu_stay ie ON di.hadm_id = ie.hadm_id AND ie.rn = 1 WHERE di.icd_version = 9 AND di.icd_code IN ('9670', '9671', '9672')
  UNION DISTINCT SELECT DISTINCT pr.subject_id, ie.stay_id FROM `physionet-data.mimiciv_hosp.procedures_icd` pr JOIN first_icu_stay ie ON pr.hadm_id = ie.hadm_id AND ie.rn = 1 WHERE pr.icd_version = 10 AND pr.icd_code IN ('5A1935Z', '5A1945Z', '5A1955Z', '0BH17EZ')
),
niv_day1 AS (
  SELECT DISTINCT ce.subject_id, ce.stay_id
  FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.stay_id IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) AND REGEXP_CONTAINS(LOWER(ce.value), r'(?i)(bipap|cpap|non[- ]invasive ventilation|niv)')
),
central_line_day1 AS (
  SELECT DISTINCT ce.subject_id, ce.stay_id FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.stay_id IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) AND REGEXP_CONTAINS(LOWER(value), r'(?i)(central line|cvc|central venous catheter|picc|ij line|subclavian line|femoral line central)')
),
arterial_line_day1 AS (
  SELECT DISTINCT ce.subject_id, ce.stay_id FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.stay_id IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) AND REGEXP_CONTAINS(LOWER(value), r'(?i)(a-?line|art line|arterial line|radial art|femoral art)')
),
dialysis_line_day1 AS (
  SELECT DISTINCT ce.subject_id, ce.stay_id FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.stay_id IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) AND REGEXP_CONTAINS(LOWER(value), r'(?i)(dialysis cath|vascath|permcath|hd cath|hemodialysis cath)')
),
evd_line_day1 AS (
  SELECT DISTINCT ce.subject_id, ce.stay_id FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.stay_id IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) AND (REGEXP_CONTAINS(LOWER(value), r'(?i)(evd|external ventricular drain|ventriculostomy)') OR ce.itemid IN (227351, 227354))
),
ecmo_line_day1 AS (
   SELECT DISTINCT pe.subject_id, pe.stay_id FROM `physionet-data.mimiciv_icu.procedureevents` pe JOIN first_icu_stay icu ON pe.stay_id = icu.stay_id AND icu.rn = 1
   WHERE pe.itemid IN (225441, 225442, 225443, 225444, 225445, 225446, 225447) AND pe.starttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR)
   UNION DISTINCT SELECT DISTINCT ce.subject_id, ce.stay_id FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
   WHERE ce.stay_id IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) AND (REGEXP_CONTAINS(LOWER(value), r'(?i)(ecmo|extracorporeal membrane oxygenation)') OR ce.itemid IN (228161, 228162))
),
rass_data_day1 AS (
   SELECT ce.stay_id, ce.valuenum, ce.charttime, ROW_NUMBER() OVER (PARTITION BY ce.stay_id ORDER BY ce.charttime ASC) as rn
   FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
   WHERE ce.itemid = 228096 AND ce.stay_id IS NOT NULL AND ce.valuenum IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR)
),
rass_summary_day1 AS (
   SELECT rd.stay_id, MAX(CASE WHEN rn = 1 THEN rd.valuenum ELSE NULL END) as first_rass_day1, MAX(CASE WHEN rd.valuenum > 0 THEN 1 ELSE 0 END) as agitated_day1, MAX(CASE WHEN rd.valuenum <= -4 THEN 1 ELSE 0 END) as deeply_sedated_day1, MIN(rd.valuenum) as min_rass_day1, MAX(rd.valuenum) as max_rass_day1
   FROM rass_data_day1 rd GROUP BY rd.stay_id
),
cam_icu_day1 AS (
  SELECT ce.stay_id, MAX(CASE WHEN LOWER(value) LIKE '%positive%' THEN 1 WHEN valuenum = 1 THEN 1 ELSE 0 END) as positive_cam_day1
  FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.itemid = 228332 AND ce.stay_id IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) GROUP BY ce.stay_id
),
pain_scores_day1 AS (
  SELECT ce.stay_id, MAX(CASE WHEN ce.itemid = 223791 THEN ce.valuenum ELSE NULL END) AS max_nrs_pain_day1, MAX(CASE WHEN ce.itemid = 228299 THEN ce.valuenum ELSE NULL END) AS max_cpot_day1, MAX(CASE WHEN ce.itemid = 223791 THEN 1 ELSE 0 END) AS nrs_recorded_day1, MAX(CASE WHEN ce.itemid = 228299 THEN 1 ELSE 0 END) AS cpot_recorded_day1
  FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.itemid IN (223791, 228299) AND ce.stay_id IS NOT NULL AND ce.valuenum IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) GROUP BY ce.stay_id
),
ciwa_scores_day1 AS (
  SELECT ce.stay_id, MAX(CASE WHEN ce.itemid = 227364 THEN ce.valuenum ELSE NULL END) AS max_ciwa_day1, MAX(CASE WHEN ce.itemid = 227364 THEN 1 ELSE 0 END) AS ciwa_recorded_day1
  FROM `physionet-data.mimiciv_icu.chartevents` ce JOIN first_icu_stay icu ON ce.stay_id = icu.stay_id AND icu.rn = 1
  WHERE ce.itemid = 227364 AND ce.stay_id IS NOT NULL AND ce.valuenum IS NOT NULL AND ce.charttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) GROUP BY ce.stay_id
),
-- Chemical Restraint Medications within FIRST 24 HOURS
chem_restraint_inputs_day1 AS (
    SELECT DISTINCT inp.stay_id
    FROM `physionet-data.mimiciv_icu.inputevents` inp
    JOIN first_icu_stay icu ON inp.stay_id = icu.stay_id AND icu.rn = 1
    WHERE inp.itemid IN (221744, 225974, 225975, 222168, 225942, 221662, 221385, 221668, 223773, 221468)
    AND inp.starttime BETWEEN icu.intime AND TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR)
),
chem_restraint_prescriptions_day1 AS (
    SELECT DISTINCT icu.stay_id
    FROM `physionet-data.mimiciv_hosp.prescriptions` pr
    JOIN first_icu_stay icu ON pr.hadm_id = icu.hadm_id AND icu.rn = 1
    WHERE (LOWER(pr.drug) LIKE '%haloperidol%' OR LOWER(pr.drug) LIKE '%haldol%' OR LOWER(pr.drug) LIKE '%droperidol%' OR LOWER(pr.drug) LIKE '%ziprasidone%' OR LOWER(pr.drug) LIKE '%geodon%' OR LOWER(pr.drug) LIKE '%olanzapine%' OR LOWER(pr.drug) LIKE '%zyprexa%' OR LOWER(pr.drug) LIKE '%ketamine%' OR LOWER(pr.drug) LIKE '%propofol%' OR LOWER(pr.drug) LIKE '%diprivan%' OR LOWER(pr.drug) LIKE '%dexmedetomidine%' OR LOWER(pr.drug) LIKE '%precedex%' OR LOWER(pr.drug) LIKE '%diazepam%' OR LOWER(pr.drug) LIKE '%valium%' OR LOWER(pr.drug) LIKE '%lorazepam%' OR LOWER(pr.drug) LIKE '%ativan%' OR LOWER(pr.drug) LIKE '%midazolam%' OR LOWER(pr.drug) LIKE '%versed%' OR LOWER(pr.drug) LIKE '%alprazolam%' OR LOWER(pr.drug) LIKE '%xanax%' OR LOWER(pr.drug) LIKE '%clonazepam%' OR LOWER(pr.drug) LIKE '%klonopin%' OR LOWER(pr.drug) LIKE '%temazepam%' OR LOWER(pr.drug) LIKE '%restoril%' OR LOWER(pr.drug) LIKE '%oxazepam%' OR LOWER(pr.drug) LIKE '%serax%' OR LOWER(pr.drug) LIKE '%diphenhydramine%' OR LOWER(pr.drug) LIKE '%benadryl%')
    -- Prescription active *during* first 24 hours
    AND pr.starttime < TIMESTAMP_ADD(icu.intime, INTERVAL 24 HOUR) AND pr.stoptime > icu.intime
),
chem_restraint_day1_stays AS (
    SELECT stay_id FROM chem_restraint_inputs_day1
    UNION DISTINCT
    SELECT stay_id FROM chem_restraint_prescriptions_day1
),

-- Severity Scores - First 24 Hours ----------------------------------------
first_day_sofa_score AS ( SELECT stay_id, MAX(sofa_24hours) as sofa_24hours FROM `physionet-data.mimiciv_derived.sofa` WHERE stay_id IS NOT NULL GROUP BY stay_id ),
first_day_oasis_score AS ( SELECT stay_id, MAX(oasis) as oasis FROM `physionet-data.mimiciv_derived.oasis` WHERE stay_id IS NOT NULL GROUP BY stay_id ),
first_day_sapsii_score AS ( SELECT stay_id, MAX(sapsii) as sapsii FROM `physionet-data.mimiciv_derived.sapsii` WHERE stay_id IS NOT NULL GROUP BY stay_id ),

-- Baseline Demographics & Admission Diagnoses -------------------------------
patient_info AS (
  -- ADDED deathtime
  SELECT DISTINCT
    pa.subject_id, pa.gender, pa.anchor_age AS age, pa.anchor_year_group,
    ad.race, ad.language, ad.admission_location, ad.hadm_id, ad.deathtime
  FROM `physionet-data.mimiciv_hosp.patients` pa
  JOIN `physionet-data.mimiciv_hosp.admissions` ad ON pa.subject_id = ad.subject_id
  JOIN first_icu_stay fis ON ad.hadm_id = fis.hadm_id AND fis.rn = 1
),
mental_disorder_pts AS (
  SELECT ie.subject_id, ie.hadm_id, ie.stay_id, MAX(CASE WHEN (di.icd_version = 10 AND REGEXP_CONTAINS(di.icd_code, r'^F2')) OR (di.icd_version = 9 AND REGEXP_CONTAINS(di.icd_code, r'^295')) THEN 1 ELSE 0 END) AS has_psychosis, MAX(CASE WHEN (di.icd_version = 10 AND REGEXP_CONTAINS(di.icd_code, r'^F30|^F31')) OR (di.icd_version = 9 AND REGEXP_CONTAINS(di.icd_code, r'^296')) THEN 1 ELSE 0 END) AS has_mania_bipolar, MAX(CASE WHEN (di.icd_version = 10 AND REGEXP_CONTAINS(di.icd_code, r'^F1')) OR (di.icd_version = 9 AND REGEXP_CONTAINS(di.icd_code, r'^303|^304')) THEN 1 ELSE 0 END) AS has_substance_use, MAX(CASE WHEN (di.icd_version = 10 AND REGEXP_CONTAINS(di.icd_code, r'^F0')) OR (di.icd_version = 9 AND REGEXP_CONTAINS(di.icd_code, r'^293')) THEN 1 ELSE 0 END) AS has_phys_condition_mental_disorder
  FROM `physionet-data.mimiciv_hosp.diagnoses_icd` di JOIN first_icu_stay ie ON di.hadm_id = ie.hadm_id AND ie.rn = 1
  WHERE ( (di.icd_version = 10 AND REGEXP_CONTAINS(di.icd_code, r'^F[0-3]')) OR (di.icd_version = 9 AND REGEXP_CONTAINS(di.icd_code, r'^29[356]|^30[34]')) ) GROUP BY ie.subject_id, ie.hadm_id, ie.stay_id
),
suicide_attempt_dx AS (
  SELECT DISTINCT ie.subject_id, ie.stay_id FROM `physionet-data.mimiciv_hosp.diagnoses_icd` di JOIN first_icu_stay ie ON di.hadm_id = ie.hadm_id AND ie.rn = 1
  WHERE (di.icd_version = 9 AND SUBSTR(di.icd_code, 1, 3) = 'E95') OR (di.icd_version = 10 AND (REGEXP_CONTAINS(di.icd_code, r'^X[6-7][0-9]') OR di.icd_code = 'X80' OR di.icd_code = 'X81' OR di.icd_code = 'X82' OR di.icd_code = 'X83' OR di.icd_code = 'X84' OR REGEXP_CONTAINS(di.icd_code, r'^T1491')))
)

-- Final SELECT: Combines all patient info for the first ICU stay -----------
SELECT
  -- Identifiers / Context
  pi.subject_id, icu.stay_id, icu.hadm_id, icu.los AS length_of_stay, icu.first_careunit AS icu_type, pi.anchor_year_group AS cohort_year,
  -- Baseline Demographics
  pi.gender, pi.age, COALESCE(pi.race, 'Unknown') AS race,
   CASE WHEN pi.admission_location LIKE '%EMERGENCY%' THEN 'ED' WHEN pi.admission_location LIKE '%TRANSFER%' THEN 'Transfer' WHEN pi.admission_location LIKE '%PHYS REFERRAL%' OR pi.admission_location LIKE '%CLINIC REFERRAL%' THEN 'Referral' WHEN pi.admission_location LIKE '%SURGERY%' OR pi.admission_location LIKE '%PACU%' THEN 'Surgery/PACU' ELSE 'Other' END AS admission_source,
  COALESCE(pi.language, 'Unknown') AS language,
  -- Baseline Admission Diagnoses
  COALESCE(md.has_psychosis, 0) AS has_psychosis_admission_dx, COALESCE(md.has_mania_bipolar, 0) AS has_mania_bipolar_admission_dx, COALESCE(md.has_substance_use, 0) AS has_substance_use_admission_dx, COALESCE(md.has_phys_condition_mental_disorder, 0) AS has_phys_condition_mental_disorder_admission_dx, CASE WHEN sad.subject_id IS NOT NULL THEN 1 ELSE 0 END AS admitted_with_suicide_attempt_dx,
  -- Clinical Status - First 24 Hours
  CASE WHEN v.subject_id IS NOT NULL THEN 1 ELSE 0 END AS ventilated_day1, CASE WHEN niv.subject_id IS NOT NULL THEN 1 ELSE 0 END AS niv_day1, CASE WHEN cl.subject_id IS NOT NULL THEN 1 ELSE 0 END AS has_cline_day1, CASE WHEN al.subject_id IS NOT NULL THEN 1 ELSE 0 END AS has_aline_day1, CASE WHEN dl.subject_id IS NOT NULL THEN 1 ELSE 0 END AS has_dline_day1, CASE WHEN evd.subject_id IS NOT NULL THEN 1 ELSE 0 END AS has_evd_day1, CASE WHEN ecmo.subject_id IS NOT NULL THEN 1 ELSE 0 END AS has_ecmo_day1,
  rsd1.first_rass_day1, rsd1.min_rass_day1, rsd1.max_rass_day1, COALESCE(rsd1.agitated_day1, 0) AS agitated_day1, COALESCE(rsd1.deeply_sedated_day1, 0) AS deeply_sedated_day1, COALESCE(camd1.positive_cam_day1, 0) AS positive_cam_day1,
  COALESCE(psd1.nrs_recorded_day1, 0) AS nrs_recorded_day1, psd1.max_nrs_pain_day1, COALESCE(psd1.cpot_recorded_day1, 0) AS cpot_recorded_day1, psd1.max_cpot_day1,
  COALESCE(csd1.ciwa_recorded_day1, 0) AS ciwa_recorded_day1, csd1.max_ciwa_day1,
  -- Chemical Restraint Use within First 24 Hours (Predictor)
  CASE WHEN crd1.stay_id IS NOT NULL THEN 1 ELSE 0 END AS chemical_restraint_day1,
  -- Severity Scores - First 24 Hours
  sofa.sofa_24hours AS first_day_sofa, oasis.oasis AS first_day_oasis, saps.sapsii AS first_day_sapsii,
  -- Outcomes / Full Stay Variables
  CASE WHEN rd.restraint_24hr_periods_count IS NOT NULL THEN 1 ELSE 0 END AS physically_restrained, -- Primary outcome: Ever physically restrained? (0 or 1)
  COALESCE(rd.restraint_24hr_periods_count, 0) AS physical_restraint_24hr_periods, -- Secondary outcome: Count of distinct 24hr periods from admission with restraint (0, 1, 2...)
  -- Death within 24h of first restraint
  COALESCE(CASE
      WHEN frt.first_restraint_time IS NOT NULL AND pi.deathtime IS NOT NULL -- Patient was restrained and died
           AND pi.deathtime >= frt.first_restraint_time -- Died on or after first restraint
           AND pi.deathtime <= TIMESTAMP_ADD(frt.first_restraint_time, INTERVAL 24 HOUR) -- Died within 24h window
      THEN 1
      ELSE 0 -- Not restrained, didn't die, or died outside the window
  END, 0) AS died_within_24h_of_first_restraint

FROM first_icu_stay icu
JOIN patient_info pi ON icu.subject_id = pi.subject_id AND icu.hadm_id = pi.hadm_id
-- Outcome related joins
LEFT JOIN restraint_periods_count rd ON icu.stay_id = rd.stay_id -- Provides the count of distinct 24hr periods with restraint
LEFT JOIN first_restraint_time_cte frt ON icu.stay_id = frt.stay_id -- Join for first restraint time
-- Baseline Admission Dx Joins
LEFT JOIN mental_disorder_pts md ON icu.stay_id = md.stay_id
LEFT JOIN suicide_attempt_dx sad ON icu.stay_id = sad.stay_id
-- Day 1 Clinical Status Joins
LEFT JOIN ventilated_day1 v ON icu.stay_id = v.stay_id
LEFT JOIN niv_day1 niv ON icu.stay_id = niv.stay_id
LEFT JOIN central_line_day1 cl ON icu.stay_id = cl.stay_id
LEFT JOIN arterial_line_day1 al ON icu.stay_id = al.stay_id
LEFT JOIN dialysis_line_day1 dl ON icu.stay_id = dl.stay_id
LEFT JOIN evd_line_day1 evd ON icu.stay_id = evd.stay_id
LEFT JOIN ecmo_line_day1 ecmo ON icu.stay_id = ecmo.stay_id
LEFT JOIN rass_summary_day1 rsd1 ON icu.stay_id = rsd1.stay_id
LEFT JOIN cam_icu_day1 camd1 ON icu.stay_id = camd1.stay_id
LEFT JOIN pain_scores_day1 psd1 ON icu.stay_id = psd1.stay_id
LEFT JOIN ciwa_scores_day1 csd1 ON icu.stay_id = csd1.stay_id
LEFT JOIN chem_restraint_day1_stays crd1 ON icu.stay_id = crd1.stay_id -- Join for Day 1 chem restraint
-- Day 1 Severity Score Joins
LEFT JOIN first_day_sofa_score sofa ON icu.stay_id = sofa.stay_id
LEFT JOIN first_day_oasis_score oasis ON icu.stay_id = oasis.stay_id
LEFT JOIN first_day_sapsii_score saps ON icu.stay_id = saps.stay_id

WHERE icu.rn = 1 AND icu.los >= 1.0; -- Only first ICU stays >= 1 day
