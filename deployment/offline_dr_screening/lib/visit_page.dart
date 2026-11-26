import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'patient_store.dart';
import 'screening_page.dart';

class VisitPage extends StatefulWidget {
  const VisitPage(
      {super.key,
      required this.patient,
      required this.visit,
      required this.store});
  final PatientRecord patient;
  final VisitRecord visit;
  final PatientStore store;

  @override
  State<VisitPage> createState() => _VisitPageState();
}

class _VisitPageState extends State<VisitPage> {
  static const _channel = MethodChannel('offirst/reports');
  bool _busy = false;

  Future<void> _report(String action) async {
    setState(() => _busy = true);
    try {
      final patient = widget.patient;
      final visit = widget.visit;
      final allExperts = await widget.store.loadExperts();
      final scope = visitScope(visit);

      final assignedExperts = allExperts
          .where((expert) => patient.expertIds.contains(expert.id))
          .toList();
      final rows = <Map<String, String>>[];
      for (final eye in Eye.values) {
        final images = visit.forEye(eye);
        for (final image in images) {
          rows.add({
            'id': image.id,
            'eye': '${eye.label} (${eye.code})',
            'path': (await widget.store.original(image)).path,
            'date': image.capturedAt.toLocal().toString().split('.').first,
            'result':
                '${(image.probability * 100).toStringAsFixed(1)}% model probability; '
                    '${image.flagged ? 'at or above' : 'below'} the '
                    '${(image.threshold * 100).toStringAsFixed(0)}% threshold.',
            'classification': image.flagged
                ? 'Model flag at or above threshold; not a diagnosis.'
                : 'Below threshold; DR is not ruled out.',
            'quality': 'Not assessed; no validated quality model is available.',
            'model': image.modelVersion,
          });
        }
      }
      final result = await _channel.invokeMethod<bool>(action, {
        'visit': visit.id,
        'heading': [
          'Report ID: OFFIRST-${visit.id}-v3',
          'Patient: ${patient.name}',
          'Patient ID: ${patient.id}',
          if (patient.age != null) 'Recorded age: ${patient.age}',
          'Visit: ${visit.id}',
          'Visit date: ${visit.date.toLocal().toString().split('.').first}',
          'Screening scope: $scope',
          'Models recorded in this visit: ${widget.visit.images.map((image) => image.modelVersion).toSet().join(', ')}',
          'Recorded threshold(s): ${widget.visit.images.map((image) => image.threshold.toStringAsPrecision(16)).toSet().join(', ')}',
          'Image-quality status: Not assessed; no validated quality model is available in this build.',
          'Input-validation status: New images are not classified while validated fundus detection is unavailable.',
          for (final expert in assignedExperts)
            'Locally associated expert (not evidence of review): ${expert.name}${expert.specialty == null ? '' : ' · ${expert.specialty}'}${expert.clinic == null ? '' : ' · ${expert.clinic}'}${expert.phone == null ? '' : ' · ${expert.phone}'}${expert.email == null ? '' : ' · ${expert.email}'}',
          'General guidance: seek qualified professional assessment for concerning findings or visual symptoms, regardless of the model score.',
          for (final eye in Eye.values) visitEyeStatus(visit, eye),
          'Interpretation: Stored classifier output is experimental and not a confirmed diagnosis. A result below threshold does not rule out diabetic retinopathy.',
          'Recommended next step: Arrange assessment by a qualified eye-care professional; seek urgent care for sudden vision loss or severe eye symptoms.',
          screeningNotice,
        ],
        'images': rows,
      });
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
    return Scaffold(
      appBar: AppBar(title: const Text('Screening visit')),
      body: ListView(padding: const EdgeInsets.all(20), children: [
        Text(widget.patient.name,
            style: Theme.of(context).textTheme.headlineSmall),
        Text(MaterialLocalizations.of(context)
            .formatMediumDate(widget.visit.date.toLocal())),
        Text('Screening scope: $scope'),
        const SizedBox(height: 12),
        for (final eye in Eye.values) Text(visitEyeStatus(widget.visit, eye)),
        for (final eye in Eye.values)
          EyeResults(
            eye: eye,
            images: widget.visit.forEye(eye),
            original: widget.store.original,
          ),
        const SizedBox(height: 12),
        const Text(screeningNotice),
        if (_busy) const LinearProgressIndicator(),
        Wrap(spacing: 8, children: [
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
      ]),
    );
  }
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
