import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offirst_dr_screening/patient_store.dart';
import 'package:offirst_dr_screening/visit_page.dart';

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
    ImageRecord image(Eye eye, double score) => ImageRecord(
          id: recordId('IMG'),
          eye: eye,
          capturedAt: date,
          path: 'test-fixture.png',
          probability: score,
          threshold: .48,
          modelVersion: 'prototype',
        );
    await tester.pumpWidget(MaterialApp(
      home: VisitPage(
        patient: PatientRecord(id: 'demo', name: 'Demo', createdAt: date),
        visit: VisitRecord(id: 'visit-1', date: date, images: [
          image(Eye.left, .49),
          image(Eye.right, .47),
        ]),
        store: PatientStore(),
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
      final rows = args['images'] as List;
      expect(
          rows.map((row) => row['eye']), ['Right eye (OD)', 'Left eye (OS)']);
      expect(rows[0]['result'], contains('below'));
      expect(rows[1]['result'], contains('at or above'));
    }
  });
}
