import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offirst_dr_screening/patient_store.dart';
import 'package:offirst_dr_screening/screening_probes.dart';
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
            age: 54,
            notes: 'Local note',
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
      await tester.scrollUntilVisible(button, 120,
          scrollable: find.byType(Scrollable).last);
      await tester.tap(button);
      await tester.pumpAndSettle();
      // Save and export confirmation snackbars overlay the lower action row.
      // Let the transient message expire before tapping the next report action.
      await tester.pump(const Duration(seconds: 5));
      await tester.pumpAndSettle();
    }

    expect(find.text('Screening scope: Bilateral (OD + OS)'), findsOneWidget);
    expect(seen.map((call) => call.method), ['save', 'view', 'export']);
    for (final call in seen) {
      final args = call.arguments as Map;
      expect(args['reportId'], 'OFFIRST-visit-1-v6');
      expect(args.containsKey('research'), isFalse);
      expect(args.containsKey('scope'), isFalse);
      expect(args.containsKey('rows'), isFalse);
      expect(args['campSite'], '—');
      expect(args['disclaimer'], contains('not a standalone'));
      expect(args['disclaimer'], isNot(contains('SAMPLE')));
      expect(_when.hasMatch(args['dateTime'] as String), isTrue);
      final patient = args['patient'] as Map;
      expect(patient['name'], 'Demo');
      expect(patient['id'], 'demo');
      expect(patient['ageSex'], '54 yrs / —');
      expect(patient['eyes'], 'Right (OD) & Left (OS)');
      _expectBlankOptionals(patient);
      final triage = args['triage'] as Map;
      expect(triage['level'], 'unavailable');
      expect(triage['badge'], 'NOT GRADED');
      expect(triage['title'], 'Screening Grade Unavailable');
      expect(args['findingsState'], 'unavailable');
      expect(args['findings'], isEmpty);
      expect(args['notFlagged'], '');
      final drEyes = args['drEyes'] as List;
      expect(drEyes.map((eye) => eye['name']),
          ['Right Eye (OD)', 'Left Eye (OS)']);
      for (final eye in drEyes) {
        _expectUngradedEye(eye as Map);
      }
      final photos = args['photos'] as List;
      expect(photos.map((photo) => photo['title']),
          ['Right Eye (OD)', 'Left Eye (OS)']);
      for (final photo in photos) {
        _expectUngradedPhoto(photo as Map, 'test-fixture.png');
      }
    }
  });

  testWidgets('unilateral report records an unavailable eye without a result',
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
    final right = ImageRecord(
      id: 'image-right',
      eye: Eye.right,
      capturedAt: date,
      path: 'right.png',
      probability: .2,
      threshold: .48,
      modelVersion: 'v1.1-test',
    );
    await tester.pumpWidget(MaterialApp(
      home: VisitPage(
        patient: PatientRecord(id: 'demo', name: 'Demo', createdAt: date),
        visit: VisitRecord(
          id: 'unilateral-visit',
          date: date,
          images: [right],
          unavailableEyes: {Eye.left},
        ),
        store: _ReportStore([]),
      ),
    ));
    final save = find.text('Save PDF');
    await tester.scrollUntilVisible(save, 120,
        scrollable: find.byType(Scrollable).last);
    await tester.tap(save);
    await tester.pumpAndSettle();

    expect(find.text('Screening scope: Unilateral (OD only)'), findsOneWidget);
    expect(
        find.textContaining('Left eye (OS): Unavailable'), findsOneWidget);
    final args = seen.single.arguments as Map;
    final patient = args['patient'] as Map;
    expect(patient['eyes'], 'Right (OD)');
    expect(patient['ageSex'], '—');
    expect(args['reportId'], 'OFFIRST-unilateral-visit-v6');
    expect(args['findingsState'], 'unavailable');
    final drEyes = args['drEyes'] as List;
    expect(drEyes, hasLength(1));
    expect(drEyes.single['name'], 'Right Eye (OD)');
    _expectUngradedEye(drEyes.single as Map);
    final photos = args['photos'] as List;
    expect(photos, hasLength(1));
    expect(photos.single['title'], 'Right Eye (OD)');
    _expectUngradedPhoto(photos.single as Map, 'right.png');
  });

  testWidgets('OS-only report does not invent a right-eye result',
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
    final left = ImageRecord(
      id: 'image-left',
      eye: Eye.left,
      capturedAt: date,
      path: 'left.png',
      probability: .6,
      threshold: .48,
      modelVersion: 'v1.1-test',
    );
    await tester.pumpWidget(MaterialApp(
      home: VisitPage(
        patient: PatientRecord(id: 'demo', name: 'Demo', createdAt: date),
        visit: VisitRecord(
          id: 'os-only-visit',
          date: date,
          images: [left],
          unavailableEyes: {Eye.right},
        ),
        store: _ReportStore([]),
      ),
    ));
    final save = find.text('Save PDF');
    await tester.scrollUntilVisible(save, 120,
        scrollable: find.byType(Scrollable).last);
    await tester.tap(save);
    await tester.pumpAndSettle();

    expect(find.text('Screening scope: Unilateral (OS only)'), findsOneWidget);
    expect(
        find.textContaining('Right eye (OD): Unavailable'), findsOneWidget);
    final args = seen.single.arguments as Map;
    final patient = args['patient'] as Map;
    expect(patient['eyes'], 'Left (OS)');
    expect(args['reportId'], 'OFFIRST-os-only-visit-v6');
    final photos = args['photos'] as List;
    expect(photos, hasLength(1));
    expect(photos.single['title'], 'Left Eye (OS)');
    _expectUngradedPhoto(photos.single as Map, 'left.png');
    expect((args['drEyes'] as List).single['name'], 'Left Eye (OS)');
  });

  testWidgets('research report is explicit about manual input confirmation',
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
    final image = ImageRecord(
      id: 'demo-image',
      eye: Eye.left,
      capturedAt: date,
      path: 'test-fixture.png',
      probability: .42,
      threshold: 0.3675977844516245,
      modelVersion: 'offirst-v2.1-experimental-43a87f05',
    );
    await tester.pumpWidget(MaterialApp(
      home: VisitPage(
        patient: PatientRecord(id: 'demo', name: 'Demo', createdAt: date),
        visit: VisitRecord(
          id: 'research-visit',
          date: date,
          images: [image],
          unavailableEyes: {Eye.right},
          researchDemonstration: true,
        ),
        store: _ReportStore([]),
      ),
    ));
    final save = find.text('Save PDF');
    await tester.scrollUntilVisible(save, 120,
        scrollable: find.byType(Scrollable).last);
    await tester.tap(save);
    await tester.pumpAndSettle();

    expect(find.textContaining('RESEARCH DEMONSTRATION ONLY'), findsOneWidget);
    final args = seen.single.arguments as Map;
    expect(args['disclaimer'], contains('Research demonstration only'));
    expect(args['disclaimer'], contains('not a standalone'));
    expect(args.containsKey('research'), isFalse);
    expect(args['reportId'], 'OFFIRST-research-visit-v6');
    final photos = args['photos'] as List;
    expect(photos, hasLength(1));
    expect(photos.single['title'], 'Left Eye (OS)');
    _expectUngradedPhoto(photos.single as Map, 'test-fixture.png');
    final triage = args['triage'] as Map;
    expect(triage['badge'], 'NOT GRADED');
  });

  testWidgets('graded visit sends screening colours for every disease',
      (tester) async {
    const channel = MethodChannel('offirst/reports');
    final seen = <MethodCall>[];
    final messenger = tester.binding.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(channel, (call) async {
      seen.add(call);
      return true;
    });
    addTearDown(() => messenger.setMockMethodCallHandler(channel, null));
    final probes = await tester.runAsync(ScreeningProbes.load);
    final embedding =
        List<double>.generate(768, (index) => index == 0 ? 1.0 : 0.0);
    final date = DateTime.utc(2026, 1, 1);
    ImageRecord image(Eye eye, String path) => ImageRecord(
          id: 'image-${eye.code}',
          eye: eye,
          capturedAt: date,
          path: path,
          probability: .2,
          threshold: .48,
          modelVersion: 'offirst-v2.1-experimental-b6bcb118',
          embedding: embedding,
        );
    await tester.pumpWidget(MaterialApp(
      home: VisitPage(
        patient: PatientRecord(id: 'demo', name: 'Demo', createdAt: date),
        visit: VisitRecord(
          id: 'graded-visit',
          date: date,
          images: [image(Eye.left, 'left.jpg'), image(Eye.right, 'right.jpg')],
        ),
        store: _ReportStore([]),
      ),
    ));
    final save = find.text('Save PDF');
    await tester.scrollUntilVisible(save, 120,
        scrollable: find.byType(Scrollable).last);
    await tester.tap(save);
    await tester.pumpAndSettle();

    final loaded = probes!;
    final severity = loaded.severity(embedding);
    final perEye = loaded.screeningFindings(embedding);
    final args = seen.single.arguments as Map;
    expect(args['reportId'], 'OFFIRST-graded-visit-v6');
    expect(args.containsKey('rows'), isFalse);
    final patient = args['patient'] as Map;
    expect(patient['ageSex'], '—');
    expect(patient['eyes'], 'Right (OD) & Left (OS)');
    final drEyes = args['drEyes'] as List;
    expect(drEyes.map((eye) => eye['name']),
        ['Right Eye (OD)', 'Left Eye (OS)']);
    final percent = '${(severity.confidence * 100).round()}%';
    final caption = severity.index == 0 ? 'Routine' : 'Primary driver';
    for (final eye in drEyes) {
      expect(eye['gradeIndex'], severity.index);
      expect(eye['score'], '${severity.label} · $percent');
      expect(eye['caption'], caption);
      expect(eye['level'],
          ['healthy', 'mild', 'moderate', 'severe'][severity.index]);
    }
    final triage = args['triage'] as Map;
    if (severity.index == 0) {
      expect(triage['level'], 'healthy');
      expect(triage['badge'], 'HEALTHY');
      expect(triage['title'], 'Action Recommended: Routine Annual Screening');
    } else {
      expect(triage['badge'], '${severity.label.toUpperCase()} DR (OD & OS)');
      expect(triage['title'], contains('Action Recommended'));
    }
    final flagged = perEye.where((finding) => finding.signal != null).toList();
    final rows = args['findings'] as List;
    if (flagged.isEmpty) {
      expect(args['findingsState'], 'clear');
      expect(rows, isEmpty);
      expect(args['notFlagged'], '');
    } else {
      expect(args['findingsState'], 'flags');
      expect(
          args['notFlagged'],
          perEye
              .where((finding) => finding.signal == null)
              .map((finding) => finding.name)
              .join(', '));
      expect(
          {
            for (final row in rows)
              '${row['name']}|${row['eye']}|${row['signal']}|${row['confidence']}',
          },
          {
            for (final finding in flagged)
              for (final eye in ['Right (OD)', 'Left (OS)'])
                '${finding.name}|$eye|${finding.signal}|${(finding.probability * 100).round()}%',
          });
      final shown = rows.map((row) => int.parse(
          (row['confidence'] as String).replaceAll('%', ''))).toList();
      for (var index = 1; index < shown.length; index++) {
        expect(shown[index], lessThanOrEqualTo(shown[index - 1]));
      }
      final eyesByName = <String, List<String>>{};
      for (final row in rows) {
        expect(row['name'], isNot('Diabetic Retinopathy'));
        expect(row['name'], isNot('Any Disease Risk'));
        expect(
            row['level'],
            switch (row['signal']) {
              'Borderline' => 'mild',
              'Moderate' => 'moderate',
              'High / Marked' => 'severe',
              _ => fail('unexpected signal ${row['signal']}'),
            });
        eyesByName
            .putIfAbsent(row['name'] as String, () => [])
            .add(row['eye'] as String);
      }
      for (final eyes in eyesByName.values) {
        expect(eyes, ['Right (OD)', 'Left (OS)']);
      }
    }
    expect(args['notFlagged'], isNot(contains('Diabetic Retinopathy')));
    expect(args['notFlagged'], isNot(contains('Any Disease Risk')));
    final photos = args['photos'] as List;
    expect(photos.map((photo) => photo['title']),
        ['Right Eye (OD)', 'Left Eye (OS)']);
    expect(photos[0]['path'], 'right.jpg');
    expect(photos[1]['path'], 'left.jpg');
    expect(photos[0]['quality'], 'Not assessed');
    final pills = photos[0]['pills'] as List;
    expect(pills.first['label'], '${severity.label} DR · $percent');
    if (flagged.isEmpty) {
      expect(pills[1]['label'], 'No Other Flags');
      expect(pills[1]['level'], 'healthy');
      expect(photos[0]['summary'],
          '${severity.label} DR; no secondary multi-disease alerts');
    } else {
      final strongest = flagged.reduce((best, finding) =>
          finding.probability > best.probability ? finding : best);
      expect(pills[1]['label'], contains(strongest.name));
      expect(photos[0]['summary'], startsWith('${severity.label} DR;'));
    }
  });
}

final _when = RegExp(
    r'^\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}, \d{1,2}:\d{2} (AM|PM)$');

void _expectBlankOptionals(Map patient) {
  expect(patient['contact'], '—');
  expect(patient['diabetes'], '—');
  expect(patient['operator'], '—');
  expect(patient['device'], '—');
  expect(patient.containsKey('notes'), isFalse);
  expect(patient.containsKey('age'), isFalse);
}

void _expectUngradedEye(Map eye) {
  expect(eye['caption'], 'Not graded');
  expect(eye['level'], 'unavailable');
  expect(eye['gradeIndex'], -1);
  expect(eye['score'], '—');
}

void _expectUngradedPhoto(Map photo, String path) {
  expect(photo['path'], path);
  expect(photo['quality'], 'Not assessed');
  expect(photo['summary'], 'Screening grade unavailable for this image.');
  final pills = photo['pills'] as List;
  expect(pills.single['label'], 'Not graded');
  expect(pills.single['level'], 'unavailable');
}
