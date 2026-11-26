import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'offirst_theme.dart';

enum TourTarget { register, patient, mode, images, save, recent, pdf, expert }

enum TourEye { right, left }

class OnboardingStep {
  const OnboardingStep({
    required this.label,
    required this.message,
    required this.tab,
    this.navSlot,
    this.target,
    this.eye,
    this.needsPatient = false,
    this.needsVisit = false,
    this.passHole = false,
    this.allowPage = false,
    this.opensVisit = false,
    this.expandsExpert = false,
  });

  final String label;
  final String message;
  final int tab;
  final int? navSlot;
  final TourTarget? target;
  final TourEye? eye;
  final bool needsPatient;
  final bool needsVisit;
  final bool passHole;
  final bool allowPage;
  final bool opensVisit;
  final bool expandsExpert;
}

const tourTabCount = 5;

const onboardingSteps = <OnboardingStep>[
  OnboardingStep(
    label: 'Home',
    message: 'Home shows the patients, visits, and images saved on this phone.',
    tab: 0,
    navSlot: 0,
  ),
  OnboardingStep(
    label: 'Patients',
    message: 'Patients keeps each local record and the visits saved for that person.',
    tab: 1,
    navSlot: 1,
  ),
  OnboardingStep(
    label: 'Screening',
    message: 'Screening captures fundus photos and runs the on-device grade.',
    tab: 2,
    navSlot: 2,
  ),
  OnboardingStep(
    label: 'Experts',
    message: 'Experts stores eye-care contacts that can be linked to a patient.',
    tab: 3,
    navSlot: 3,
  ),
  OnboardingStep(
    label: 'About',
    message: 'About explains what this offline screening build does and its limits.',
    tab: 4,
    navSlot: 4,
  ),
  OnboardingStep(
    label: 'Add a patient',
    message: 'Add a patient to continue.',
    tab: 1,
    target: TourTarget.register,
    needsPatient: true,
    passHole: true,
  ),
  OnboardingStep(
    label: 'Choose a patient',
    message: 'Choose the patient for this screening visit.',
    tab: 2,
    target: TourTarget.patient,
    allowPage: true,
  ),
  OnboardingStep(
    label: 'Screening mode',
    message: 'Choose DR severity or multi-disease screening.',
    tab: 2,
    target: TourTarget.mode,
    allowPage: true,
  ),
  OnboardingStep(
    label: 'Right eye',
    message: 'Add the right-eye (OD) fundus image.',
    tab: 2,
    target: TourTarget.images,
    eye: TourEye.right,
    allowPage: true,
  ),
  OnboardingStep(
    label: 'Left eye',
    message: 'Add the left-eye (OS) fundus image.',
    tab: 2,
    target: TourTarget.images,
    eye: TourEye.left,
    allowPage: true,
  ),
  OnboardingStep(
    label: 'Save the visit',
    message: 'Finalize and save the visit to continue.',
    tab: 2,
    target: TourTarget.save,
    needsVisit: true,
    allowPage: true,
  ),
  OnboardingStep(
    label: 'Recent visit',
    message: 'Open the most recent visit.',
    tab: 0,
    target: TourTarget.recent,
    opensVisit: true,
    passHole: true,
  ),
  OnboardingStep(
    label: 'Visit PDF',
    message: 'Save, view, or export the visit PDF from here.',
    tab: 0,
    target: TourTarget.pdf,
    passHole: true,
  ),
  OnboardingStep(
    label: 'Associated expert',
    message: 'Link an associated expert to this patient.',
    tab: 1,
    target: TourTarget.expert,
    expandsExpert: true,
    allowPage: true,
  ),
];

int tourIndex(TourTarget target) =>
    onboardingSteps.indexWhere((step) => step.target == target);

abstract class OnboardingStore {
  Future<bool> isComplete();
  Future<void> markComplete();
}

/// Remembers a finished or skipped tour in app-private storage.
class FileOnboardingStore implements OnboardingStore {
  FileOnboardingStore({Future<Directory> Function()? directory})
      : _directory = directory ?? getApplicationSupportDirectory;
  final Future<Directory> Function() _directory;

  Future<File> get _file async =>
      File('${(await _directory()).path}/onboarding_complete_v2');

  @override
  Future<bool> isComplete() async => (await _file).exists();

  @override
  Future<void> markComplete() async {
    final file = await _file;
    await file.parent.create(recursive: true);
    await file.writeAsString('1', flush: true);
  }
}

/// In-memory flag for tests.
class MemoryOnboardingStore implements OnboardingStore {
  MemoryOnboardingStore({bool complete = false}) : complete = complete;
  bool complete;

  @override
  Future<bool> isComplete() async => complete;

  @override
  Future<void> markComplete() async {
    complete = true;
  }
}

class OnboardingOverlay extends StatelessWidget {
  const OnboardingOverlay({
    super.key,
    required this.label,
    required this.message,
    required this.index,
    required this.count,
    required this.last,
    required this.nextEnabled,
    required this.onNext,
    required this.onSkip,
    this.targetKey,
    this.navigationKey,
    this.navigationSlot,
    this.passHole = false,
    this.allowPage = false,
  });

  final String label;
  final String message;
  final int index;
  final int count;
  final bool last;
  final bool nextEnabled;
  final VoidCallback onNext;
  final VoidCallback onSkip;
  final GlobalKey? targetKey;
  final GlobalKey? navigationKey;
  final int? navigationSlot;
  final bool passHole;
  final bool allowPage;

  @override
  Widget build(BuildContext context) {
    final hole = _hole(context);
    final size = MediaQuery.sizeOf(context);
    final cardWidth = math.min(340.0, size.width - 32);
    final anchor = hole?.center.dx ?? size.width / 2;
    final left =
        (anchor - cardWidth / 2).clamp(16.0, size.width - 16 - cardWidth);
    double? top;
    double? bottom;
    if (hole == null) {
      bottom = 96;
    } else if (size.height - hole.bottom > 180) {
      top = hole.bottom + 12;
    } else {
      bottom = size.height - hole.top + 12;
    }
    return Stack(children: [
      Positioned.fill(child: _barrier(context, hole)),
      Positioned(
        left: left,
        top: top,
        bottom: bottom,
        width: cardWidth,
        child: Material(
          color: OffirstColors.background,
          elevation: 8,
          borderRadius: BorderRadius.circular(8),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(label,
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: OffirstColors.text,
                        fontWeight: FontWeight.w700)),
                const SizedBox(height: 6),
                Text(message,
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: OffirstColors.secondaryText, height: 1.35)),
                const SizedBox(height: 12),
                Row(children: [
                  TextButton(onPressed: onSkip, child: const Text('Skip')),
                  const Spacer(),
                  Text('${index + 1} of $count',
                      style: Theme.of(context).textTheme.labelMedium?.copyWith(
                          color: OffirstColors.secondaryText)),
                  const SizedBox(width: 12),
                  FilledButton(
                    onPressed: nextEnabled ? onNext : null,
                    child: Text(last ? 'Done' : 'Next'),
                  ),
                ]),
              ],
            ),
          ),
        ),
      ),
    ]);
  }

  Widget _barrier(BuildContext context, Rect? hole) {
    final paint = IgnorePointer(
      child: CustomPaint(
        painter: _TourDimPainter(hole),
        child: const SizedBox.expand(),
      ),
    );
    if (allowPage) {
      final navigation = _box(context, navigationKey);
      return Stack(children: [
        paint,
        if (navigation != null)
          Positioned.fromRect(rect: navigation, child: _block()),
      ]);
    }
    if (!passHole || hole == null) {
      return GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () {},
        child: paint,
      );
    }
    return Stack(children: [
      paint,
      Positioned(
          left: 0, top: 0, right: 0, height: math.max(0, hole.top), child: _block()),
      Positioned(
          left: 0,
          top: hole.bottom,
          right: 0,
          bottom: 0,
          child: _block()),
      Positioned(
          left: 0,
          top: hole.top,
          width: math.max(0, hole.left),
          height: hole.height,
          child: _block()),
      Positioned(
          left: hole.right,
          top: hole.top,
          right: 0,
          height: hole.height,
          child: _block()),
    ]);
  }

  Widget _block() => GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: () {},
        child: const SizedBox.expand(),
      );

  Rect? _hole(BuildContext context) {
    final slot = navigationSlot;
    if (slot != null) {
      final navigation = _box(context, navigationKey);
      if (navigation == null || navigation.width <= 0) return null;
      final width = navigation.width / tourTabCount;
      return _clamp(
          Rect.fromLTWH(navigation.left + width * slot, navigation.top, width,
              navigation.height),
          MediaQuery.sizeOf(context));
    }
    return _clamp(_box(context, targetKey), MediaQuery.sizeOf(context));
  }

  Rect? _box(BuildContext context, GlobalKey? key) {
    final overlay = context.findRenderObject();
    final target = key?.currentContext?.findRenderObject();
    if (overlay is! RenderBox ||
        target is! RenderBox ||
        !overlay.hasSize ||
        !target.hasSize ||
        target.size.isEmpty) {
      return null;
    }
    final origin = overlay.globalToLocal(target.localToGlobal(Offset.zero));
    return origin & target.size;
  }

  Rect? _clamp(Rect? hole, Size size) {
    if (hole == null) return null;
    final clipped = hole.intersect(Offset.zero & size);
    if (clipped.width <= 0 || clipped.height <= 0) return null;
    return clipped;
  }
}

class _TourDimPainter extends CustomPainter {
  const _TourDimPainter(this.hole);
  final Rect? hole;

  @override
  void paint(Canvas canvas, Size size) {
    final shade = Path()..addRect(Offset.zero & size);
    RRect? spot;
    if (hole != null) {
      spot = RRect.fromRectAndRadius(
        hole!.inflate(6).intersect(Offset.zero & size),
        const Radius.circular(12),
      );
      shade
        ..addRRect(spot)
        ..fillType = PathFillType.evenOdd;
    }
    canvas.drawPath(shade, Paint()..color = const Color(0xB312161F));
    if (spot != null) {
      canvas.drawRRect(
        spot,
        Paint()
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..color = OffirstColors.cyan,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _TourDimPainter oldDelegate) =>
      oldDelegate.hole != hole;
}
