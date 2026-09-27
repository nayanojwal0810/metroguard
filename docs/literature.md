# Literature Reviewed

This document summarizes the technical literature reviewed during the engineering and design of the MetroGuard system.

| Literature | Topic reviewed | Relevance |
|---|---|---|
| Davari, N., Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2021). *Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry.* IEEE DSAA 2021. DOI: `10.1109/DSAA53316.2021.9564181`. | Unsupervised anomaly detection on railway compressor telemetry using autoencoder reconstruction error. | Technical reference for sparse autoencoder anomaly detection on compressor sensor telemetry and failure interval detection requirements ($\ge 2$ hours). |
| Veloso, B., Ribeiro, R. P., Pereira, P. M., & Gama, J. (2022). *The MetroPT dataset for predictive maintenance.* Nature Scientific Data 9, 764. DOI: `10.1038/s41597-022-01877-3`. | Data collection architecture, sensor specifications, and failure event annotation protocol for metro train APUs. | Technical reference for sensor measurement physics (pressure, current, temperature), failure modes, and interval-overlap evaluation protocols. |
| Barros, M., Veloso, B., Pereira, P. M., Ribeiro, R. P., & Gama, J. (2020). *Failure detection of an air production unit in operational context.* Springer IoT Streams, pp. 61–74. DOI: `10.1007/978-3-030-66770-2_5`. | Rule-based and early ML detection of compressor air leaks. | Background on operational baseline behaviors and failure manifestation patterns. |
| UCI Machine Learning Repository (2020). *MetroPT-3 Dataset.* DOI: `10.24432/C5VW3R`. | Primary multivariate time-series dataset containing 1,516,948 sensor observations across 15 features. | Authoritative repository metadata source for the 2020 compressor telemetry dataset used in this project. |
