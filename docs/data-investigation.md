# Data Investigation (Read Only)

The source archive is `data/synthea_sample_data_fhir_latest.tar.xz`. Investigation used Python's standard JSON parser and extracted into a temporary directory with:

```sh
mkdir -p /tmp/hwai-fhir-investigation
tar -xJf data/synthea_sample_data_fhir_latest.tar.xz -C /tmp/hwai-fhir-investigation --strip-components=1
```

## Shape and scale

- 109 patient bundles and 2 shared bundles.
- Patient bundle sizes: 108,255 bytes to 52,345,312 bytes; median 2,325,159 bytes. 97 bundles are at least 1 MB, 9 at least 5 MB, and 4 at least 10 MB.
- Patient entry counts: 47 to 18,551 (total 118,868 resources).
- Most common patient resource types: Observation 46,305; Procedure 16,063; DiagnosticReport 11,013; Claim 9,493; ExplanationOfBenefit 9,493; Encounter 5,635; DocumentReference 5,635; MedicationRequest 3,858; Condition 3,540; AllergyIntolerance 105.
- Shared hospital bundle has 557 entries (279 Locations, 278 Organizations); shared practitioner bundle has 556 (278 Practitioners, 278 PractitionerRoles).

## Representative patients

- Smallest: `Stanton715_Schimmel440_ee4b7339-ca58-b6af-c199-04b6d5761c73.json`, 108 KB and 47 resources.
- Largest: `Cole117_Corwin846_bca1691f-8839-1d66-ed01-471134d55738.json`, 52.35 MB and 18,551 resources, including 9,610 Observations and 1,566 each Claim/EOB.
- Challenge example: `Merlene950_Marlin805_Thompson596_f5749532-3295-ad01-d9b5-932c997e7a01.json`, 2.21 MB and 622 resources. It has eight HbA1c (LOINC 4548-4) readings from 2016–2025; latest is 6.31% on 2025-09-29, Observation id `f5749532-3295-ad01-c588-daa4b0203574`. An active prediabetes Condition is present (id suffix `fdfb-caeae3913925`, onset 2000-11-20). It has no AllergyIntolerance resource.

## Findings that affect design

- **Notes are represented twice.** There are 5,635 DocumentReferences and 5,635 DiagnosticReports with `presentedForm`. Across all patients, the ordered payloads are byte-identical between the two representations. Total decoded payload size counting both is about 15.2 MB; the largest individual payload is 10,352 bytes. Canonicalize duplicate text for context/retrieval while retaining both source IDs and their relationship.
- **Status matters.** DocumentReferences are 5,526 superseded and 109 current (one current per patient); MedicationRequests are 3,590 completed and 268 active; CareTeams and CarePlans also mix active/inactive or active/completed. All Encounters are finished, and Observations/DiagnosticReports are final in this data.
- **Missing allergy data is common.** There are 105 AllergyIntolerance records across 109 patients; a bundle with no such resource cannot support a definitive “no allergies” claim.
- **Reference forms differ.** No broken `urn:uuid:` references were found within patient bundles. Shared Practitioner/Organization/Location references use identifier-query references, not local bundle UUIDs; 745 distinct such references occur 122,715 times.
- **Date meanings vary.** Observation effective dates range 1960–2026-08-16; Condition onset dates range 1947–2026-08-16. Keep effective, issued, recorded, onset, and abatement dates distinct.
- **Billing is large and paired.** Claim and ExplanationOfBenefit each contribute 9,493 resources. Defer these from default context and only index them if eval/product needs justify the cost.

The inventory is descriptive of this synthetic corpus only; it does not establish clinical importance or behavior on real records.
