{{- define "heart-api.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "heart-api.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "heart-api.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "heart-api.labels" -}}
app.kubernetes.io/name: {{ include "heart-api.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end -}}
