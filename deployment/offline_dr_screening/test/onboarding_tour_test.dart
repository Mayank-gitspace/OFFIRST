import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:offirst_dr_screening/main.dart';
import 'package:offirst_dr_screening/onboarding_tour.dart';
import 'package:offirst_dr_screening/patient_store.dart';
import 'package:offirst_dr_screening/screening_service.dart';

class _EmptyStore extends PatientStore {
  @override
  Future<List<PatientRecord>> load() async => [];

  @override
  Future<List<ExpertRecord>> loadExperts() async => [];
}

class _ReadyStore extends PatientStore {
  @override
  Future<List<PatientRecord>> load() async => [
        PatientRecord(
          id: 'OFF-tour',
          name: 'Tour Patient',
          createdAt: DateTime.utc(2026, 1, 1),
          visits: [
            VisitRecord(
              id: 'VIS-tour',
              date: DateTime.utc(2026, 2, 2),
              images: const [],
            ),
          ],
        ),
      ];

  @override
  Future<List<ExpertRecord>> loadExperts() async => [];

  @override
  Future<File> original(ImageRecord image) async => File(image.path);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('completion flag stays on disk', () async {
    final directory = await Directory.systemTemp.createTemp('offirst-tour');
    addTearDown(() => directory.delete(recursive: true));
    final store = FileOnboardingStore(directory: () async => directory);
    expect(await store.isComplete(), isFalse);
    await store.markComplete();
    expect(await store.isComplete(), isTrue);
    expect(
        await File('${directory.path}/onboarding_complete_v2').readAsString(),
        '1');
  });

  testWidgets('tab introduction continues to the add-patient gate',
      (tester) async {
    final store = MemoryOnboardingStore();
    await tester.pumpWidget(_app(_EmptyStore(), store));
    await tester.pump();

    expect(
        find.text(
            'Home shows the patients, visits, and images saved on this phone.'),
        findsOneWidget);
    expect(find.text('Skip'), findsOneWidget);
    expect(find.text('Next'), findsOneWidget);
    expect(find.text('1 of ${onboardingSteps.length}'), findsOneWidget);

    await _next(tester);
    expect(
        find.text(
            'Patients keeps each local record and the visits saved for that person.'),
        findsOneWidget);
    expect(find.text('Patients'), findsWidgets);

    await _next(tester);
    expect(
        find.text(
            'Screening captures fundus photos and runs the on-device grade.'),
        findsOneWidget);

    await _next(tester);
    expect(
        find.text(
            'Experts stores eye-care contacts that can be linked to a patient.'),
        findsOneWidget);

    await _next(tester);
    expect(
        find.text(
            'About explains what this offline screening build does and its limits.'),
        findsOneWidget);
    expect(find.text('Done'), findsNothing);

    await _next(tester);
    expect(find.text('Add a patient to continue.'), findsOneWidget);
    expect(
        tester
            .widget<FilledButton>(find.widgetWithText(FilledButton, 'Next'))
            .onPressed,
        isNull);
    await tester.pump(const Duration(seconds: 1));
  });

  testWidgets('a saved patient and visit finish the screening workflow',
      (tester) async {
    final store = MemoryOnboardingStore();
    await tester.pumpWidget(_app(_ReadyStore(), store));
    await tester.pump();

    const messages = [
      'Home shows the patients, visits, and images saved on this phone.',
      'Patients keeps each local record and the visits saved for that person.',
      'Screening captures fundus photos and runs the on-device grade.',
      'Experts stores eye-care contacts that can be linked to a patient.',
      'About explains what this offline screening build does and its limits.',
      'Add a patient to continue.',
      'Choose the patient for this screening visit.',
      'Choose DR severity or multi-disease screening.',
      'Add the right-eye (OD) fundus image.',
      'Add the left-eye (OS) fundus image.',
      'Finalize and save the visit to continue.',
      'Open the most recent visit.',
    ];
    for (final message in messages) {
      expect(find.text(message), findsOneWidget);
      expect(
          tester
              .widget<FilledButton>(find.widgetWithText(FilledButton, 'Next'))
              .onPressed,
          isNotNull);
      if (message.contains('right-eye')) {
        expect(
            tester
                .widget<SegmentedButton<Eye>>(
                    find.byType(SegmentedButton<Eye>))
                .selected,
            {Eye.right});
      }
      if (message.contains('left-eye')) {
        expect(
            tester
                .widget<SegmentedButton<Eye>>(
                    find.byType(SegmentedButton<Eye>))
                .selected,
            {Eye.left});
      }
      await _next(tester);
    }

    expect(find.text('Save, view, or export the visit PDF from here.'),
        findsOneWidget);
    expect(find.text('Save PDF'), findsOneWidget);
    expect(find.text('View PDF'), findsOneWidget);
    expect(find.text('Export PDF'), findsOneWidget);
    expect(find.text('Tour Patient'), findsWidgets);

    await _next(tester);
    expect(find.text('Link an associated expert to this patient.'),
        findsOneWidget);
    expect(find.text('Locally associated experts'), findsOneWidget);
    expect(find.text('Done'), findsOneWidget);
    expect(find.text('${onboardingSteps.length} of ${onboardingSteps.length}'),
        findsOneWidget);

    await tester.tap(find.widgetWithText(FilledButton, 'Done'));
    await tester.pump();
    await tester.pump();
    await tester.pump(const Duration(seconds: 1));
    expect(find.text('Skip'), findsNothing);
    expect(store.complete, isTrue);

    await tester.pumpWidget(_app(_ReadyStore(), store));
    await tester.pump();
    expect(find.text('Skip'), findsNothing);
    expect(find.text('Done'), findsNothing);
  });

  testWidgets('skip saves completion and hides the tour', (tester) async {
    final store = MemoryOnboardingStore();
    await tester.pumpWidget(_app(_EmptyStore(), store));
    await tester.pump();
    await tester.tap(find.text('Skip'));
    await tester.pump();
    await tester.pump();
    expect(find.text('Next'), findsNothing);
    expect(store.complete, isTrue);

    await tester.pumpWidget(_app(_EmptyStore(), store));
    await tester.pump();
    expect(find.text('Skip'), findsNothing);
  });
}

Future<void> _next(WidgetTester tester) async {
  await tester.tap(find.widgetWithText(FilledButton, 'Next'));
  await tester.pump();
  await tester.pump(const Duration(milliseconds: 600));
}

Widget _app(PatientStore patients, OnboardingStore onboarding) =>
    OffirstScreeningApp(
      store: patients,
      onboarding: onboarding,
      screeningService: ScreeningService.forTesting(
        initializeModel: () async {},
        runPrediction: (_) async => const ScreeningResult(.1),
      ),
    );
