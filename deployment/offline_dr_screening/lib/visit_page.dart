import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'onboarding_tour.dart';
import 'patient_store.dart';
import 'screening_page.dart';
import 'screening_probes.dart';

class VisitPage extends StatefulWidget {
  const VisitPage(
      {super.key,
      required this.patient,
      required this.visit,
      required this.store,
      this.tourStep,
      this.onTourNext,
      this.onTourSkip});
  final PatientRecord patient;
  final VisitRecord visit;
  final PatientStore store;
  final ValueNotifier<int?>? tourStep;
  final VoidCallback? onTourNext;
  final VoidCallback? onTourSkip;

  @override
  State<VisitPage> createState() => _VisitPageState();
}

class _VisitPageState extends State<VisitPage> {
  static const _channel = MethodChannel('offirst/reports');
  final _pdfKey = GlobalKey();
  bool _busy = false;
  ScreeningMode _mode = ScreeningMode.severity;
  ScreeningProbes? _probes;
  Object? _probeError;

  bool get _hasEmbedding =>
      widget.visit.images.any((image) => image.embedding != null);

  @override
  void initState() {
    super.initState();
    widget.tourStep?.addListener(_revealPdf);
    if (_hasEmbedding) unawaited(_loadProbes());
    WidgetsBinding.instance.addPostFrameCallback((_) => _revealPdf());
  }

  @override
  void dispose() {
    widget.tourStep?.removeListener(_revealPdf);
    super.dispose();
  }

  void _revealPdf() {
    final step = widget.tourStep?.value;
    if (step == null || onboardingSteps[step].target != TourTarget.pdf) return;
    final target = _pdfKey.currentContext;
    if (target != null) {
      Scrollable.ensureVisible(target, alignment: 0.5);
    }
  }

  Future<void> _loadProbes() async {
    try {
      final probes = await ScreeningProbes.load();
      if (mounted) setState(() => _probes = probes);
    } catch (error) {
      if (mounted) setState(() => _probeError = error);
    }
  }

  Future<void> _report(String action) async {
    setState(() => _busy = true);
    try {
      final patient = widget.patient;
      final visit = widget.visit;
      final when = _reportWhen(visit.date.toLocal());
      final probes = _hasEmbedding ? await _probesForReport() : null;
      final captured = <_ReportEye>[];
      for (final eye in _pdfEyes) {
        final image = _latestImage(visit, eye);
        if (image == null) continue;
        SeverityGrade? severity;
        var findings = const <ScreeningFinding>[];
        final embedding = image.embedding;
        if (embedding != null && probes != null) {
          try {
            severity = probes.severity(embedding);
            findings = probes.screeningFindings(embedding);
          } catch (_) {
            severity = null;
            findings = const [];
          }
        }
        captured.add(_ReportEye(
          eye: eye,
          path: (await widget.store.original(image)).path,
          severity: severity,
          findings: findings,
        ));
      }
      final result = await _channel.invokeMethod<bool>(
          action, _reportPayload(patient, visit, when, captured));
      if (mounted && result == true && action != 'view') {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text(
              action == 'export' ? 'PDF exported.' : 'PDF saved in OFFIRST.'),
        ));
      }
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text(
              'Could not complete the PDF action. To open the file, install a PDF viewer or export it.'),
        ));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final scope = visitScope(widget.visit);
    final tour = widget.tourStep;
    return Stack(children: [
      Scaffold(
      appBar: AppBar(title: const Text('Screening visit')),
      body: ListView(padding: const EdgeInsets.all(20), children: [
        Text(widget.patient.name,
            style: Theme.of(context).textTheme.headlineSmall),
        Text(MaterialLocalizations.of(context)
            .formatMediumDate(widget.visit.date.toLocal())),
        Text('Screening scope: $scope'),
        const SizedBox(height: 12),
        if (_hasEmbedding) ...[
          ScreeningModeSelector(
            mode: _mode,
            onChanged: (mode) => setState(() => _mode = mode),
          ),
          const SizedBox(height: 12),
        ],
        for (final eye in Eye.values) Text(visitEyeStatus(widget.visit, eye)),
        for (final eye in Eye.values)
          EyeResults(
            eye: eye,
            images: widget.visit.forEye(eye),
            original: widget.store.original,
            mode: _mode,
            probes: _probes,
            probesFailed: _probeError != null,
          ),
        const SizedBox(height: 12),
        if (widget.visit.researchDemonstration)
          const Text(
              'RESEARCH DEMONSTRATION ONLY — Non-diagnostic; not for clinical decisions. Input was manually confirmed, not automatically validated.'),
        if (!widget.visit.researchDemonstration) const Text(screeningNotice),
        if (_busy) const LinearProgressIndicator(),
        KeyedSubtree(
          key: _pdfKey,
          child: Wrap(spacing: 8, children: [
            TextButton(
                onPressed: _busy ? null : () => _report('save'),
                child: const Text('Save PDF')),
            TextButton(
                onPressed: _busy ? null : () => _report('view'),
                child: const Text('View PDF')),
            TextButton(
                onPressed: _busy ? null : () => _report('export'),
                child: const Text('Export PDF')),
          ]),
        ),
      ]),
      ),
      if (tour != null)
        ValueListenableBuilder<int?>(
          valueListenable: tour,
          builder: (context, step, _) {
            if (step == null ||
                onboardingSteps[step].target != TourTarget.pdf) {
              return const SizedBox.shrink();
            }
            final current = onboardingSteps[step];
            return OnboardingOverlay(
              label: current.label,
              message: current.message,
              index: step,
              count: onboardingSteps.length,
              last: false,
              nextEnabled: true,
              targetKey: _pdfKey,
              passHole: true,
              onNext: widget.onTourNext ?? () {},
              onSkip: widget.onTourSkip ?? () {},
            );
          },
        ),
    ]);
  }

  Future<ScreeningProbes?> _probesForReport() async {
    if (_probes != null) return _probes;
    try {
      return await ScreeningProbes.load();
    } catch (_) {
      return null;
    }
  }

}

const _pdfEyes = [Eye.right, Eye.left];

const _reportDisclaimer =
    'AI-assisted screening summary for triage only — not a standalone medical diagnosis. Consult a qualified ophthalmologist.';

class _ReportEye {
  const _ReportEye({
    required this.eye,
    required this.path,
    required this.severity,
    required this.findings,
  });

  final Eye eye;
  final String path;
  final SeverityGrade? severity;
  final List<ScreeningFinding> findings;

  List<ScreeningFinding> get flagged =>
      findings.where((finding) => finding.signal != null).toList();
}

class _FlagRow {
  const _FlagRow(this.eye, this.finding);
  final Eye eye;
  final ScreeningFinding finding;
}

ImageRecord? _latestImage(VisitRecord visit, Eye eye) {
  ImageRecord? latest;
  for (final image in visit.forEye(eye)) {
    if (latest == null || image.capturedAt.isAfter(latest.capturedAt)) {
      latest = image;
    }
  }
  return latest;
}

String _reportWhen(DateTime local) {
  const months = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ];
  final hour = local.hour % 12 == 0 ? 12 : local.hour % 12;
  final minute = local.minute.toString().padLeft(2, '0');
  final suffix = local.hour >= 12 ? 'PM' : 'AM';
  return '${local.day} ${months[local.month - 1]} ${local.year}, $hour:$minute $suffix';
}

String _pct(double value) => '${(value * 100).round()}%';

String _eyeTitle(Eye eye) =>
    '${eye == Eye.right ? 'Right' : 'Left'} Eye (${eye.code})';

String _eyeShort(Eye eye) =>
    '${eye == Eye.right ? 'Right' : 'Left'} (${eye.code})';

String _gradeLevel(int index) => switch (index) {
      0 => 'healthy',
      1 => 'mild',
      2 => 'moderate',
      3 => 'severe',
      _ => 'unavailable',
    };

String _signalLevel(String signal) => switch (signal) {
      'Borderline' => 'mild',
      'Moderate' => 'moderate',
      'High / Marked' => 'severe',
      _ => 'healthy',
    };

String _signalShort(String signal) =>
    signal == 'High / Marked' ? 'High' : signal;

String _eyesCaptured(VisitRecord visit) {
  final captured = [
    for (final eye in _pdfEyes)
      if (visit.forEye(eye).isNotEmpty) _eyeShort(eye),
  ];
  return captured.isEmpty ? '—' : captured.join(' & ');
}

Map<String, Object?> _reportPayload(PatientRecord patient, VisitRecord visit,
    String when, List<_ReportEye> captured) {
  final scored = [
    for (final eye in captured)
      if (eye.severity != null) eye,
  ];
  var worst = -1;
  for (final eye in scored) {
    if (eye.severity!.index > worst) worst = eye.severity!.index;
  }
  final flags = <_FlagRow>[
    for (final eye in captured)
      for (final finding in eye.flagged) _FlagRow(eye.eye, finding),
  ]..sort((a, b) {
      final byProbability =
          b.finding.probability.compareTo(a.finding.probability);
      if (byProbability != 0) return byProbability;
      final byEye = _pdfEyes.indexOf(a.eye).compareTo(_pdfEyes.indexOf(b.eye));
      if (byEye != 0) return byEye;
      return a.finding.name.compareTo(b.finding.name);
    });
  final notFlagged = <String>[];
  if (scored.isNotEmpty) {
    for (final finding in scored.first.findings) {
      final flagged = scored.any((eye) => eye.findings.any((item) =>
          item.code == finding.code && item.signal != null));
      if (!flagged) notFlagged.add(finding.name);
    }
  }
  final state = scored.isEmpty
      ? 'unavailable'
      : flags.isEmpty
          ? 'clear'
          : 'flags';
  return {
    'visit': visit.id,
    'reportId': 'OFFIRST-${visit.id}-v6',
    'dateTime': when,
    'campSite': '—',
    'disclaimer': visit.researchDemonstration
        ? 'Research demonstration only. $_reportDisclaimer'
        : _reportDisclaimer,
    'patient': {
      'name': patient.name,
      'id': patient.id,
      'ageSex': patient.age == null ? '—' : '${patient.age} yrs / —',
      'contact': '—',
      'diabetes': '—',
      'eyes': _eyesCaptured(visit),
      'operator': '—',
      'device': '—',
    },
    'triage': _triage(scored, worst, flags.isNotEmpty),
    'drEyes': [
      for (final eye in captured)
        {
          'name': _eyeTitle(eye.eye),
          'caption': _caption(eye.severity?.index ?? -1,
              eye.severity != null && eye.severity!.index == worst && worst > 0),
          'level': eye.severity == null
              ? 'unavailable'
              : _gradeLevel(eye.severity!.index),
          'gradeIndex': eye.severity?.index ?? -1,
          'score': eye.severity == null
              ? '—'
              : '${eye.severity!.label} · ${_pct(eye.severity!.confidence)}',
        },
    ],
    'findingsState': state,
    'findings': [
      for (final row in flags)
        {
          'name': row.finding.name,
          'eye': _eyeShort(row.eye),
          'signal': row.finding.signal,
          'level': _signalLevel(row.finding.signal!),
          'confidence': _pct(row.finding.probability),
        },
    ],
    'notFlagged': state == 'flags' ? notFlagged.join(', ') : '',
    'photos': [
      for (final eye in captured) _photo(eye),
    ],
  };
}

Map<String, String> _triage(
    List<_ReportEye> scored, int worst, bool extraFlags) {
  if (scored.isEmpty || worst < 0) {
    return {
      'level': 'unavailable',
      'title': 'Screening Grade Unavailable',
      'subtitle':
          'A colour grade needs a stored embedding from the current screening model.',
      'badge': 'NOT GRADED',
    };
  }
  final drivers =
      scored.where((eye) => eye.severity!.index == worst).toList();
  final where = drivers.map((eye) => _eyeTitle(eye.eye)).join(' and ');
  final codes = drivers.map((eye) => eye.eye.code).join(' & ');
  final tail = extraFlags ? ', with secondary multi-disease flags.' : '.';
  return switch (worst) {
    3 => {
        'level': 'severe',
        'title':
            'Action Recommended: Urgent Ophthalmologist Referral (Within 1–2 Weeks)',
        'subtitle':
            'Vision-threatening diabetic retinopathy signs detected in $where$tail',
        'badge': 'SEVERE DR ($codes)',
      },
    2 => {
        'level': 'moderate',
        'title':
            'Action Recommended: Ophthalmologist Exam Within 1 Month',
        'subtitle':
            'Referable diabetic retinopathy changes detected in $where$tail',
        'badge': 'MODERATE DR ($codes)',
      },
    1 => {
        'level': 'mild',
        'title': 'Action Recommended: Follow-up Within 3–6 Months',
        'subtitle':
            'Early diabetic retinopathy signs detected in $where$tail',
        'badge': 'MILD DR ($codes)',
      },
    _ => {
        'level': 'healthy',
        'title': 'Action Recommended: Routine Annual Screening',
        'subtitle': extraFlags
            ? 'No referable diabetic retinopathy grade. Other screening flags are listed below.'
            : 'No referable diabetic retinopathy grade on the captured images.',
        'badge': 'HEALTHY',
      },
  };
}

String _caption(int index, bool primary) {
  if (index < 0) return 'Not graded';
  if (primary) return 'Primary driver';
  return switch (index) {
    0 => 'Routine',
    1 => 'Follow up',
    2 => 'Referable',
    _ => 'Urgent',
  };
}

Map<String, Object?> _photo(_ReportEye eye) {
  final severity = eye.severity;
  final flagged = eye.flagged.toList()
    ..sort((a, b) => b.probability.compareTo(a.probability));
  final pills = <Map<String, String>>[];
  if (severity == null) {
    pills.add({'label': 'Not graded', 'level': 'unavailable'});
  } else {
    pills.add({
      'label': '${severity.label} DR · ${_pct(severity.confidence)}',
      'level': _gradeLevel(severity.index),
    });
    if (flagged.isEmpty) {
      pills.add({'label': 'No Other Flags', 'level': 'healthy'});
    } else {
      for (final finding in flagged.take(4)) {
        pills.add({
          'label': '${finding.name} · ${_signalShort(finding.signal!)}',
          'level': _signalLevel(finding.signal!),
        });
      }
      if (flagged.length > 4) {
        pills.add({'label': '+${flagged.length - 4}', 'level': 'unavailable'});
      }
    }
  }
  final summary = severity == null
      ? 'Screening grade unavailable for this image.'
      : flagged.isEmpty
          ? '${severity.label} DR; no secondary multi-disease alerts'
          : '${severity.label} DR; ${flagged.map((finding) => '${finding.name} (${_signalShort(finding.signal!)})').join(', ')}';
  return {
    'title': _eyeTitle(eye.eye),
    'path': eye.path,
    'quality': 'Not assessed',
    'summary': summary,
    'pills': pills,
  };
}

String visitScope(VisitRecord visit) {
  final screenedEyes =
      Eye.values.where((eye) => visit.forEye(eye).isNotEmpty).toList();
  return switch (screenedEyes.length) {
    2 => 'Bilateral (OD + OS)',
    1 => 'Unilateral (${screenedEyes.single.code} only)',
    _ => 'No eye images available; no DR assessment',
  };
}

String visitEyeStatus(VisitRecord visit, Eye eye) {
  final count = visit.forEye(eye).length;
  if (count > 0) return '${eye.label} (${eye.code}): $count image(s) recorded';
  if (visit.unavailableEyes.contains(eye)) {
    return '${eye.label} (${eye.code}): Unavailable (recorded for this visit; no image supplied)';
  }
  return '${eye.label} (${eye.code}): Not assessed (no image supplied)';
}
