import 'package:flutter/material.dart';
import 'patient_store.dart';
import 'screening_page.dart';
import 'visit_page.dart';

void main() => runApp(const OffirstScreeningApp());

class OffirstScreeningApp extends StatelessWidget {
  const OffirstScreeningApp({super.key, this.store});
  final PatientStore? store;
  @override
  Widget build(BuildContext context) => MaterialApp(
      title: 'OFFIRST',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF4F7FA),
        colorScheme: ColorScheme.fromSeed(
            seedColor: const Color(0xFF1769AA), surface: Colors.white),
        cardTheme: CardThemeData(
            color: Colors.white,
            elevation: 0,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18),
                side: const BorderSide(color: Color(0xFFE1E8EF)))),
      ),
      home: OffirstHome(store: store ?? PatientStore()));
}

class OffirstHome extends StatefulWidget {
  const OffirstHome({super.key, required this.store});
  final PatientStore store;
  @override
  State<OffirstHome> createState() => _OffirstHomeState();
}

class _OffirstHomeState extends State<OffirstHome> {
  int _tab = 0;
  PatientStore get _store => widget.store;
  late Future<List<PatientRecord>> _patients = _store.load();
  void _refresh() => setState(() => _patients = _store.load());
  Future<void> _addPatient() async {
    final name = TextEditingController();
    final age = TextEditingController();
    final ok = await showDialog<bool>(
        context: context,
        builder: (c) => AlertDialog(
                title: const Text('Register patient'),
                content: Column(mainAxisSize: MainAxisSize.min, children: [
                  TextField(
                      controller: name,
                      decoration:
                          const InputDecoration(labelText: 'Patient name')),
                  TextField(
                      controller: age,
                      keyboardType: TextInputType.number,
                      decoration:
                          const InputDecoration(labelText: 'Age (optional)'))
                ]),
                actions: [
                  TextButton(
                      onPressed: () => Navigator.pop(c, false),
                      child: const Text('Cancel')),
                  FilledButton(
                      onPressed: () => Navigator.pop(c, true),
                      child: const Text('Save'))
                ]));
    final patientName = name.text.trim();
    final rawAge = age.text.trim();
    final patientAge = int.tryParse(rawAge);
    if (ok != true || !mounted) return;
    if (patientName.isEmpty ||
        (rawAge.isNotEmpty &&
            (patientAge == null || patientAge < 0 || patientAge > 130))) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Enter a name and a valid age, or leave age blank.')));
      return;
    }
    try {
      await _store.addPatient(PatientRecord(
          id: recordId('OFF'),
          name: patientName,
          createdAt: DateTime.now().toUtc(),
          age: patientAge));
      if (mounted) _refresh();
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not save the patient.')));
      }
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
      appBar: AppBar(
        title: const Text('OFFIRST'),
        actions: const [
          Padding(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Chip(
                  avatar: Icon(Icons.wifi_off, size: 16),
                  label: Text('Offline')))
        ],
      ),
      body: IndexedStack(index: _tab, children: [
        _dashboard(),
        _patientList(),
        ScreeningPage(store: _store, patients: _patients, onSaved: _refresh),
        _experts(),
        _profile()
      ]),
      bottomNavigationBar: NavigationBar(
          selectedIndex: _tab,
          onDestinationSelected: (v) => setState(() => _tab = v),
          destinations: const [
            NavigationDestination(
                icon: Icon(Icons.home_outlined), label: 'Home'),
            NavigationDestination(
                icon: Icon(Icons.people_outline), label: 'Patients'),
            NavigationDestination(
                icon: Icon(Icons.remove_red_eye_outlined), label: 'Screening'),
            NavigationDestination(
                icon: Icon(Icons.medical_services_outlined), label: 'Experts'),
            NavigationDestination(
                icon: Icon(Icons.person_outline), label: 'Profile')
          ]));
  Widget _dashboard() => FutureBuilder<List<PatientRecord>>(
      future: _patients,
      builder: (c, s) {
        final count = s.data?.length ?? 0;
        return ListView(padding: const EdgeInsets.all(20), children: [
          const Text('Screen anywhere.',
              style: TextStyle(fontSize: 30, fontWeight: FontWeight.w700)),
          const Text('Track both eyes. Keep care moving, even offline.',
              style: TextStyle(color: Colors.blueGrey)),
          const SizedBox(height: 20),
          Card(
              child: Padding(
                  padding: const EdgeInsets.all(20),
                  child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text('OFFLINE SCREENING',
                            style: TextStyle(
                                letterSpacing: 1.2, color: Color(0xFF1769AA))),
                        const SizedBox(height: 8),
                        const Text(
                            'Experimental diabetic retinopathy screening on this device.',
                            style: TextStyle(fontSize: 18)),
                        const SizedBox(height: 14),
                        FilledButton.icon(
                            onPressed: () => setState(() => _tab = 2),
                            icon: const Icon(Icons.add_a_photo_outlined),
                            label: const Text('Start screening'))
                      ]))),
          Row(children: [
            Expanded(child: _stat('$count', 'Patients')),
            Expanded(
                child: _stat(
                    '${s.data?.fold<int>(0, (n, p) => n + p.visits.length) ?? 0}',
                    'Saved visits'))
          ]),
          const SizedBox(height: 16),
          FilledButton.tonalIcon(
              onPressed: _addPatient,
              icon: const Icon(Icons.person_add_alt),
              label: const Text('Register patient')),
          const SizedBox(height: 24),
          const _ResearchCard()
        ]);
      });
  Widget _stat(String n, String label) => Card(
      child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(children: [
            Text(n,
                style:
                    const TextStyle(fontSize: 27, fontWeight: FontWeight.bold)),
            Text(label)
          ])));
  Widget _patientList() => FutureBuilder<List<PatientRecord>>(
      future: _patients,
      builder: (c, s) {
        if (s.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (s.hasError) {
          return const Center(child: Text('Could not load patient records.'));
        }
        final rows = s.data ?? [];
        return ListView(padding: const EdgeInsets.all(16), children: [
          Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
            const Text('Patients',
                style: TextStyle(fontSize: 25, fontWeight: FontWeight.bold)),
            IconButton(
                onPressed: _addPatient, icon: const Icon(Icons.person_add))
          ]),
          if (rows.isEmpty)
            const Padding(
                padding: EdgeInsets.all(24),
                child: Text('No local patients yet. Add a profile to begin.')),
          ...rows.map((p) => Card(
                  child: ExpansionTile(
                      leading: const CircleAvatar(child: Icon(Icons.person)),
                      title: Text(p.name),
                      subtitle:
                          Text('${p.id} · ${p.visits.length} saved visits'),
                      children: [
                    if (p.age != null) ListTile(title: Text('Age ${p.age}')),
                    ...p.visits.map((v) => ListTile(
                          leading: const Icon(Icons.history),
                          title: Text(MaterialLocalizations.of(context)
                              .formatMediumDate(v.date.toLocal())),
                          subtitle: Text(
                              'OD: ${v.forEye(Eye.right).length} · OS: ${v.forEye(Eye.left).length}'),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: () => Navigator.of(context).push(
                              MaterialPageRoute<void>(
                                  builder: (_) => VisitPage(
                                      patient: p, visit: v, store: _store))),
                        ))
                  ])))
        ]);
      });
  Widget _experts() => const _InfoPage(
      icon: Icons.medical_services_outlined,
      title: 'Experts',
      body:
          'Expert directory and secure case submission are planned. No cases or images are transmitted in this offline prototype.');
  Widget _profile() => const _InfoPage(
      icon: Icons.shield_outlined,
      title: 'Privacy & technology',
      body:
          'Basic screening runs locally. Patient records are stored in app-private local storage. OFFIRST explores contrastive learning to learn useful retinal representations from less labelled data; this installed DR model is an experimental classifier and does not learn from patient images.');
}

class _InfoPage extends StatelessWidget {
  const _InfoPage(
      {required this.icon, required this.title, required this.body});
  final IconData icon;
  final String title, body;
  @override
  Widget build(BuildContext context) => Center(
      child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(icon, size: 48, color: const Color(0xFF1769AA)),
            const SizedBox(height: 12),
            Text(title,
                style:
                    const TextStyle(fontSize: 26, fontWeight: FontWeight.bold)),
            const SizedBox(height: 12),
            Text(body, textAlign: TextAlign.center)
          ])));
}

class _ResearchCard extends StatelessWidget {
  const _ResearchCard();
  @override
  Widget build(BuildContext context) => const Card(
      child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
              'Learn more from less data. OFFIRST research includes BYOL and masked contrastive representation learning. The deployed DR model is an experimental screening pathway; it is not clinically validated.',
              style: TextStyle(height: 1.45))));
}
