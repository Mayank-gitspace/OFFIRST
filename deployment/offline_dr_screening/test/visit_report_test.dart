import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offirst_dr_screening/patient_store.dart';
import 'package:offirst_dr_screening/visit_page.dart';

class _ReportStore extends PatientStore {
  _ReportStore(this.experts);
  final List<ExpertRecord> experts;

  @override
  Future<List<ExpertRecord>> loadExperts() async => experts;

  @override
  Future<File> original(ImageRecord image) async => File(image.path);
}

void main() {
  testWidgets('bilateral visit sends separate OD and OS data to PDF actions',
      (tester) async {
    const channel = MethodChannel('offirst/reports');
    final seen = <MethodCall>[];
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(channel, (call) async {
      seen.add(call);
      return true;
    });
    addTearDown(() => messenger.setMockMethodCallHandler(channel, null));

    final date = DateTime.utc(2026, 1, 1);
    final expert = ExpertRecord(
      id: 'expert-1',
      name: 'Synthetic Expert',
      updatedAt: date,
      specialty: 'Retina',
      phone: '+10000000000',
    );
    ImageRecord image(Eye eye, double score, double threshold, String model) =>
        ImageRecord(
          id: recordId('IMG'),
          eye: eye,
          capturedAt: date,
          path: 'test-fixture.png',
          probability: score,
          threshold: threshold,
          modelVersion: model,
        );
    await tester.pumpWidget(MaterialApp(
      home: VisitPage(
        patient: PatientRecord(
            id: 'demo',
            name: 'Demo',
            createdAt: date,
            expertIds: const ['expert-1']),
        visit: VisitRecord(id: 'visit-1', date: date, images: [
          image(Eye.left, .37, 0.3675977844516245,
              'offirst-v2.1-experimental-43a87f05'),
          image(Eye.right, .47, .48, 'offirst-dr-onnx-b8fc1019'),
        ]),
        store: _ReportStore([expert]),
      ),
    ));
    expect(find.text('Right eye (OD)'), findsOneWidget);
    expect(find.text('Left eye (OS)'), findsOneWidget);
    for (final label in ['Save PDF', 'View PDF', 'Export PDF']) {
      final button = find.text(label);
      await tester.ensureVisible(button);
      await tester.tap(button);
      await tester.pumpAndSettle();
    }

    expect(seen.map((call) => call.method), ['save', 'view', 'export']);
    for (final call in seen) {
      final args = call.arguments as Map;
      final headings = args['heading'] as List;
      expect(headings, contains('Right eye (OD): 1 image(s)'));
      expect(headings, contains('Left eye (OS): 1 image(s)'));
      expect(
          headings.any((line) =>
              line.toString().startsWith('Models recorded in this visit: ')),
          isTrue);
      expect(
          headings.any((line) => line
              .toString()
              .contains('Synthetic Expert · Retina · +10000000000')),
          isTrue);
      expect(
          headings.any((line) =>
              line.toString().startsWith('Recorded threshold(s): ') &&
              line.toString().contains('0.3675977844516245')),
          isTrue);
      expect(
          headings,
          contains(
              'General guidance: seek qualified professional assessment for concerning findings or visual symptoms, regardless of the model score.'));
      final rows = args['images'] as List;
      expect(
          rows.map((row) => row['eye']), ['Right eye (OD)', 'Left eye (OS)']);
      expect(rows[0]['result'], contains('below'));
      expect(rows[1]['result'], contains('at or above'));
      expect(rows[0]['model'], 'offirst-dr-onnx-b8fc1019');
      expect(rows[0]['result'], contains('48% threshold'));
      expect(rows[1]['model'], 'offirst-v2.1-experimental-43a87f05');
      expect(rows[1]['result'], contains('37% threshold'));
    }
  });
}
